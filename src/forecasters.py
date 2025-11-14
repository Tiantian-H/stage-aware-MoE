import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from torchdiffeq import odeint_adjoint, odeint
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
randname = str(random.randint(0, 1000))

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, is_pure_nn, is_candidates, attention_type='original', state_dim=68, scaling_version=0):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented
        self.is_pure_nn = is_pure_nn
        self.is_candidates = is_candidates
        self.attention_type = attention_type
        self.state_dim = state_dim
        self.scaling_version = scaling_version
        #self.alp = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.last_res_phy = None
        self.last_res_aug = None
        self.weights_phy = None
        self.weights_aug = None

        aug_expert = ['GAE', 'MLP']
        num_experts = len(aug_expert) + 1
        
        # Select attention mechanism based on config
        if attention_type == 'original':
            self.temporal_attention = TemporalAttention(hidden_dim=4, num_experts=num_experts)
        elif attention_type == 'state_aware':
            self.temporal_attention = StateAwareTemporalAttention(state_dim=state_dim, hidden_dim=8, num_experts=num_experts)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.lambda_reg = 0.1
        self.orth_loss = 0

        self.layer_norm = nn.LayerNorm(68)
        # Initialize scaling parameters based on version
        self._init_scaling_parameters()

    def _init_scaling_parameters(self):
        """Initialize scaling parameters based on scaling version"""
        if self.scaling_version == 0:
            # Version 0 (Default): Single scale/bias for combined output
            self.scalar = nn.Parameter(torch.full((1,), 1.0, dtype=torch.float32))
            self.beta = nn.Parameter(torch.full((1,), 1e-3, dtype=torch.float32))
            raise ValueError(f"Unknown scaling_version: {self.scaling_version}")

    def _apply_scaling_and_attention(self, res_normalized, attention_weights):
        """Apply scaling and attention based on scaling version"""
        if self.scaling_version == 0:
            # Version 0 (Default): Attention first, then single scaling
            res = torch.bmm(res_normalized, attention_weights.unsqueeze(2)).squeeze(2)
            res = res * self.scalar + self.beta
            
        return res

    def forward(self, t, state):
        if self.is_augmented and self.is_pure_nn == False:
            res_phy = self.model_phy(t, state)
            res_aug = self.model_aug(t, state)
            res_stack = torch.cat([res_phy, res_aug], dim=-1)
            # res_normalized = res_stack/(res_stack.norm(p=2, dim=1, keepdim=True)+1e-2)
            res_normalized = (res_stack-res_stack.mean(dim=-1, keepdim=True)) \
                            / (res_stack.std(dim=1, keepdim=True) + 1)
            self.orth_loss += self.compute_ortho_loss(res_stack)
            # Get attention weights based on attention type
            if self.attention_type == 'original':
                attention_weights = self.temporal_attention(t)
            elif self.attention_type in ['state_aware']:
                # These attention types need state information
                attention_weights = self.temporal_attention(t, state)
            else:
                attention_weights = self.temporal_attention(t)
            
            # Apply scaling and attention based on version
            res = self._apply_scaling_and_attention(res_normalized, attention_weights)

            if torch.isnan(res_phy + res_aug).any():
                print('nan inside')
                self.print_params()
            self.last_res_phy = res_phy
            self.last_res_aug = res_aug
            return res
        
        elif self.is_augmented and self.is_pure_nn == True:
            res_aug = self.model_aug(t, state)
            res_aug.unsqueeze(-1)
            self.last_res_aug = res_aug
            if torch.isnan(res_aug).any():
                print('nan inside')
                self.print_params()
                # print the component with nan
                #print('res_aug:', res_aug)
                self._reset_weights()
                return torch.zeros_like(res_aug) if self.last_res_aug is None else self.last_res_aug
            return res_aug
        
        else:
            #print('******** res_phy only *******')
            res_phy = self.model_phy(t, state).squeeze(2)
            self.last_res_phy = res_phy
            if torch.isnan(res_phy).any():
                print('nan inside')
                self.print_params()
            return res_phy

    def compute_ortho_loss(self, expert_output):
        ortho_loss = 0.0
        num_experts = expert_output.shape[-1]
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                f_i = expert_output[..., i] - expert_output[..., i].mean()
                f_j = expert_output[..., j] - expert_output[..., j].mean()
                dot = torch.sum(f_i * f_j)
                ortho_loss += dot ** 2
        return self.lambda_reg * ortho_loss

    def _reset_weights(self):
        print('Resetting weights')
        def reset_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.reset_parameters()
        self.apply(reset_weights)
    # def print params
    def print_params(self):
        if self.model_phy != 'none':
            print('phy params:', self.model_phy.params)
        if self.model_aug != 'none':
            print('aug params:', self.model_aug.params)


class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, is_pure_nn, is_candidates, method='rk4', options=None, attention_type='original', state_dim=68, scaling_version=0):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented
        self.is_pure_nn = is_pure_nn
        self.is_candidates = is_candidates
        self.attention_type = attention_type
        self.state_dim = state_dim
        self.scaling_version = scaling_version
        self.derivative_estimator = DerivativeEstimator(self.model_phy,
                                                        self.model_aug,
                                                        is_augmented= self.is_augmented,
                                                        is_pure_nn = self.is_pure_nn,
                                                        is_candidates = self.is_candidates,
                                                        attention_type = self.attention_type,
                                                        state_dim = self.state_dim,
                                                        scaling_version = self.scaling_version)
        self.method = method
        self.options = options
        self.int_ = odeint

        # for orth loss


    def forward(self, t_span, y0):
        # note: y_whole only needed for physcial models
        # y0 = y_whole[:,0,:,0] # [1,68]
        # y0 shape in forcaster torch.Size([1, 150, 68, 1])
        y0 = y0.to(device)
        t_span = t_span.to(device)
        if torch.min(t_span) > 1e-3:
            t_span = torch.cat([torch.tensor([0]).to(device), t_span.to(device)])
            res = self.int_(lambda t, y: self.derivative_estimator(t, y.to(device)),
                            y0=y0.to(device), t=t_span.to(device), rtol=1e-8, atol=1e-8)[1:]

        else:
            res = self.int_(lambda t, y: self.derivative_estimator(t, y.to(device)),
                            y0=y0.to(device), t=t_span.to(device), rtol=1e-8, atol=1e-8)

        #res: T x batch_size x n_c (x h x w)
        dim_seq = y0.dim() + 1
        dims = [1, 2, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)   # batch_size x n_c x T (x h x w)


    def get_pde_params(self):
        if not self.is_augmented and self.model_phy !='none':
            phy_org_alpha = self.model_phy.params_org.alpha_org
            phy_org_k = self.model_phy.params_org.k_org
            phy_org_c = self.model_phy.params_org.c_org
            returned_param_phy_dict = {'alpha': torch.square(phy_org_alpha), 'k': torch.square(phy_org_k), 'c': phy_org_c}
            return returned_param_phy_dict
        
    def get_detail_deriv(self):
        saved_results = {
            'res_phy': self.derivative_estimator.last_res_phy,
            'res_aug': self.derivative_estimator.last_res_aug,
            'combined': None
        }
        # If both res_phy and res_aug exist, compute and store their combined value
        if saved_results['res_phy'] is not None and saved_results['res_aug'] is not None:
            saved_results['combined'] = saved_results['res_phy'] + saved_results['res_aug']
        
        return saved_results

        
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)  # 3 experts
        )

    def forward(self, t):
        # Create time encoding
        t = t.reshape(-1, 1)
        # Get attention weights dependent on time
        weights = self.time_mlp(t)
        attention = torch.softmax(weights, dim=-1)

        return attention


class StateAwareTemporalAttention(nn.Module):
    """
    Simplified state-aware temporal attention for better performance
    """
    def __init__(self, state_dim, hidden_dim, num_experts, use_dataset_stats=False):
        super().__init__()
        self.state_dim = state_dim
        self.num_experts = num_experts
        
        # Simplified architecture for better performance
        self.time_encoder = nn.Linear(1, hidden_dim // 2)
        self.state_encoder = nn.Linear(state_dim, hidden_dim // 2)
        
        # Simple combination layer
        self.attention_head = nn.Sequential(
            nn.Linear(hidden_dim, num_experts),
            nn.ReLU()
        )
        
    def forward(self, t, x):
        batch_size = x.shape[0]
        
        # Encode time and state efficiently
        t = t.reshape(-1, 1)
        time_features = torch.relu(self.time_encoder(t))
        
        # Use mean pooling for state to reduce computation
        state_mean = x.mean(dim=-1, keepdim=True) if x.dim() > 1 else x
        state_features = torch.relu(self.state_encoder(state_mean.expand(-1, self.state_dim)))
        
        # Simple concatenation and attention
        combined = torch.cat([time_features, state_features], dim=-1)
        logits = self.attention_head(combined)
        attention = F.softmax(logits, dim=-1)
        
        return attention


