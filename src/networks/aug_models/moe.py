import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.graph_utils import graph_connectomes, dense_adj_to_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATConv, TransformerConv


class GAE_nograph(nn.Module):
    def __init__(self, cfg, state_c, edge_index, edge_attr):
        self.hid_dim=cfg.model.hidden_dim
        super().__init__()
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Linear(state_c, self.hid_dim),
            nn.ReLU(True),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(True),
            nn.Linear(self.hid_dim, state_c),
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, t, x):
        x = x.squeeze(-1)
        return self.net(x)

    def get_derivatives(self, t, x):
        batch_size, nc, T = x.shape  # 64, 68, 271
        x = x.permute(0, 2, 1)
        x = x.view(batch_size * T, nc)
        x = self.forward(t, x)
        x = x.view(batch_size, T, self.state_c)
        x = x.permute(0, 2, 1)
        return x

class GAE(MessagePassing):
    def __init__(self, cfg, state_c, edge_index, edge_attr):
        self.hid_dim=cfg.model.hidden_dim
        super(GAE, self).__init__(aggr='add')
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Linear(state_c, self.hid_dim),
            nn.ReLU(True),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(True),
            nn.Linear(self.hid_dim, state_c),
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, t, x):
        x = self.propagate(self.edge_index, x=x)
        x = x.squeeze(-1)
        return self.net(x)

class GCN(MessagePassing):
    def __init__(self, cfg, state_c, edge_index, edge_attr):
        self.hid_dim=cfg.model.hidden_dim
        super(GCN, self).__init__(aggr='add')
        self.state_c = state_c
        self.lin_in = nn.Linear(1, self.hid_dim)
        self.lin_out = nn.Linear(self.hid_dim, 1)
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, t, x):
        x = self.lin_in(x)
        x = self.propagate(self.edge_index, x=x)
        x = self.lin_out(x).squeeze(-1)
        return x


class MLP(nn.Module):
    def __init__(self, cfg, state_c):
        self.input_dim = cfg.model.input_dim
        self.hid_dim = cfg.model.hidden_dim
        super().__init__()
        self.state_c = state_c
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.ReLU(True),
            nn.Linear(self.hid_dim, self.input_dim),
        )

    def forward(self, t, x):
        return self.net(x)

    def get_derivatives(self, t, x):
        batch_size, nc, T = x.shape  # 64, 68, 271
        x = x.permute(0, 2, 1)
        x = x.view(batch_size * T, nc)
        x = self.forward(t, x)
        x = x.view(batch_size, T, self.state_c)
        x = x.permute(0, 2, 1)
        return x
