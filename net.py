import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool




class Critic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim,hidden_dim,bias=False)

    def forward(self, node_ft, graph_ft):
        h = self.linear(graph_ft)
        # h = torch.matmul(pool, self.weight)
        return torch.sum(node_ft*h, dim = 1)

class DGI(nn.Module):
    def __init__(self, gnn, critic):
        super().__init__()
        self.gnn = gnn
        self.critic = critic 
        self.loss = nn.BCEWithLogitsLoss()
        self.pool = global_mean_pool

    def pool(self,node_ft,batch):
        return self.pool(node_ft, batch)

class RegressionModel(nn.Module):
    """
    Performs logistic regression
    """
    def __init__(self, in_dim, num_task,gnn, info_max=False) -> None:
        super().__init__()
        self.gnn = gnn
        self.linear = nn.Linear(in_dim,num_task)
        self.pool = global_mean_pool
        if info_max:
            self.critic = Critic(in_dim)

    def forward(self,batch):
        h_n = self.gnn(batch.x, batch.edge_index)
        h = self.pool(h_n,batch.batch)
        
        return self.linear(h), h_n, h



