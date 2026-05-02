import torch
from graph_aug.graph_aug_cuda import drop_nodes_batch_forward, mask_nodes_batch_forward, permute_edges_batch_forward
from utils.data.dataset import GraphBatch

def _offsets(graphs):
    sizes = [g.num_nodes for g in graphs]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return torch.tensor(offsets, device=graphs[0].x.device)

def _ptr_from_sizes(sizes, device):
    ptr = [0]
    for s in sizes:
        ptr.append(ptr[-1] + s)
    return torch.tensor(ptr, device=device, dtype=torch.long)

def drop_nodes_batch(graphs, aug_ratio):
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])
    node_ptr = _ptr_from_sizes([g.num_nodes for g in graphs], x_all.device)

    x_aug, edge_index_aug, batch_aug = drop_nodes_batch_forward(
        x_all, edge_index_all, batch_vec, node_ptr, aug_ratio
    )

    return GraphBatch(x=x_aug, edge_index=edge_index_aug, batch=batch_aug)


def mask_nodes_batch(graphs, aug_ratio):
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])
    node_ptr = _ptr_from_sizes([g.num_nodes for g in graphs], x_all.device)

    x_aug = mask_nodes_batch_forward(x_all, batch_vec, node_ptr, aug_ratio)

    return GraphBatch(x=x_aug, edge_index=edge_index_all, batch=batch_vec)


def permute_edges_batch(graphs, aug_ratio):
    edge_index_all = torch.cat([g.edge_index + offset for g, offset in zip(graphs, _offsets(graphs))], dim=1).contiguous()
    edge_ptr = _ptr_from_sizes([g.edge_index.size(1) for g in graphs], edge_index_all.device)

    edge_index_aug = permute_edges_batch_forward(edge_index_all, edge_ptr, aug_ratio)
    x_all = torch.cat([g.x for g in graphs], dim=0).contiguous()
    batch_vec = torch.cat([torch.full((g.num_nodes,), i, device=g.x.device) for i, g in enumerate(graphs)])

    return GraphBatch(x=x_all, edge_index=edge_index_aug, batch=batch_vec)

def relabel_edge_index(edge_index, batch_vec):
    unique_batch = batch_vec.unique()
    node_ptr = torch.cumsum(torch.bincount(batch_vec), dim=0)
    node_ptr = torch.cat([torch.tensor([0], device=batch_vec.device), node_ptr])

    mapping = -torch.ones(batch_vec.size(0), dtype=torch.long, device=batch_vec.device)
    for i in unique_batch:
        mask = (batch_vec == i)
        idx = mask.nonzero(as_tuple=True)[0]
        mapping[idx] = torch.arange(idx.size(0), device=batch_vec.device)

    edge_index_new = mapping[edge_index]
    return edge_index_new


def graph_views_batch(all_graphs, aug, aug_ratio=0.1):
    if aug == 'dropN':
        g_aug = drop_nodes_batch(all_graphs, aug_ratio)
    elif aug == 'permE':
        g_aug = permute_edges_batch(all_graphs, aug_ratio)
    elif aug == 'maskN':
        g_aug = mask_nodes_batch(all_graphs, aug_ratio)
    else:
        raise NotImplementedError(f'{aug} not implemented')
    return g_aug
