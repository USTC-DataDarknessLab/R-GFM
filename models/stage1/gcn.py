import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_conv_layers, dropout, epsilon=None):
        super(GCN, self).__init__()
        self.global_pool = global_add_pool
        self.layer_num = num_conv_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden))
        for _ in range(num_conv_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.t2 = torch.nn.Linear(hidden, hidden)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, x, edge_index, batch):
        h = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.t2(h)
        graph_emb = self.global_pool(h, batch)
        return graph_emb
