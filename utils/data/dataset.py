import torch


class GraphBatch:
    def __init__(self, x, edge_index, batch):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
    
    def to(self, device, non_blocking=False):
        return GraphBatch(
            x=self.x.to(device, non_blocking=non_blocking),
            edge_index=self.edge_index.to(device, non_blocking=non_blocking),
            batch=self.batch.to(device, non_blocking=non_blocking)
        )

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs_list, graphs_index_list):
        self.graphs_list = graphs_list
        self.graphs_index_list = graphs_index_list

    def __len__(self):
        return len(self.graphs_index_list)

    def __getitem__(self, idx):
        l, r = self.graphs_index_list[idx]
        return self.graphs_list[l:r], idx

def graph_collate_fn(batch):
    all_graphs = []
    center_index_ranges = []
    offset = 0
    center_ids = []

    for graph_list, center_id in batch:
        num = len(graph_list)
        all_graphs.extend(graph_list)
        center_index_ranges.append((offset, offset + num))
        offset += num
        center_ids.append(center_id)
    return all_graphs, center_index_ranges, center_ids


def graph_collate_fn_with_features(global_x):
    def collate_fn(batch):
        all_graphs = []
        center_index_ranges = []
        offset = 0
        center_ids = []

        for graph_list, center_id in batch:
            graphs_with_features = []
            for g in graph_list:
                x = global_x[g.global_n_id]
                g_new = g.clone()
                g_new.x = x
                graphs_with_features.append(g_new)

            num = len(graphs_with_features)
            all_graphs.extend(graphs_with_features)
            center_index_ranges.append((offset, offset + num))
            offset += num
            center_ids.append(center_id)

        return all_graphs, center_index_ranges, center_ids

    return collate_fn
