import torch
from torch_geometric.nn import global_add_pool, FAConv

class FAGCN(torch.nn.Module):

    def __init__(self, num_features, hidden, num_conv_layers, dropout, epsilon):
        super(FAGCN, self).__init__()
        self.global_pool = global_add_pool
        self.eps = epsilon
        self.layer_num = num_conv_layers
        self.dropout = dropout
        self.hidden_dim = hidden

        self.layers = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden, epsilon, dropout))

        self.t1 = torch.nn.Linear(num_features, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, x, edge_index, batch, method = 'sum'):
        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))
        h = torch.dropout(h, p=self.dropout, train=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, raw, edge_index)
        h = self.t2(h)
        if method == 'sum':
            graph_emb = self.global_pool(h, batch)
        else:
            graph_emb = self.center_weighted_pool(h, batch)
            
        
        return graph_emb
    
    
    def center_weighted_pool(self, h, batch):
        num_graphs = batch.max().item() + 1
        graph_emb_list = []

        for i in range(num_graphs):
            mask = (batch == i)
            h_i = h[mask]
            if h_i.size(0) == 0:
                continue

            h_center = h_i[0]
            import torch.nn.functional as F
            sims = F.cosine_similarity(h_i, h_center.unsqueeze(0), dim=1)
            weights = F.softmax(sims, dim=0)

            weighted_sum = torch.sum(h_i * weights.unsqueeze(1), dim=0)
            fused = weighted_sum + h_center
            graph_emb_list.append(fused)

        graph_emb = torch.stack(graph_emb_list, dim=0)
        return graph_emb
