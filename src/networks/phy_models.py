import torch
import torch.nn as nn
from collections import OrderedDict
from src.networks.graph_utils import prepLaplacian

class NDMParamODE(nn.Module):
    def __init__(self, is_complete=False, real_params=None, capacity=None):
        super().__init__()
        self.capacity = capacity
        self.data_folder = './data/multi_connectomes/'
        self.L = torch.tensor(prepLaplacian(self.data_folder + 'tractography.csv'),
        self.is_complete = is_complete
        self.real_params = real_params
        self.params_org = nn.ParameterDict({
            'k_org': nn.Parameter(torch.tensor(0.05)),
            'alpha_org': nn.Parameter(torch.tensor(0.05)),
            'c_org': nn.Parameter(torch.tensor(1.5))  # nn.Parameter(torch.tensor(1.)),
        })
        self.params = OrderedDict()
        if self.real_params is not None:
            self.params.update(real_params)

    # c, t, x_node
    def forward(self, t, x):

        L_batch = self.L.repeat(x.shape[0], 1, 1)
        # Implement the ODE: dx/dt = -k * L @ x + alpha * x * (c - x)
        # The full model includes the ODE; a simplified model might exclude some terms
        if len(x.shape) != 3:
            x = x.unsqueeze(2)

        if self.is_complete == True:
            if self.real_params is None:
                self.params['k'] = self.params_org['k_org']
                self.params['alpha'] = self.params_org['alpha_org']
                self.params['c'] = self.params_org['c_org']
            (k, alpha, c) = list(self.params.values())
            dxdt = - k * torch.bmm(L_batch, x) + alpha * x * (c - x)
        else:
            if self.real_params is None:
                self.params['k'] = self.params_org['k_org']
            k = list(self.params.values())[0]
            dxdt = - k * torch.bmm(L_batch, x)
        return dxdt
