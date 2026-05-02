import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph
from tqdm import tqdm
import random
from typing import Tuple
def induced_graphs_multi_hop(data, k_hop=5):
    hop_range=(1, k_hop)
    all_subgraphs = []
    node_to_subgraph_range = []

    for index in tqdm(range(data.x.size(0)), desc="Generating node-based subgraphs", unit="node"):
        current_label = data.y[index].item()
        start_idx = len(all_subgraphs)

        for hop in range(hop_range[0], hop_range[1] + 1):
            min_size, max_size = hop * (hop + 1), (hop + 1) * (hop + 2)

            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=hop, edge_index=data.edge_index)

            if len(subset) < min_size:
                need_node_num = min_size - len(subset)
                pos_nodes = torch.argwhere(data.y == int(current_label))
                candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
                if candidate_nodes.shape[0] > need_node_num:
                    candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
                subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

            if len(subset) > max_size:
                subset = subset[torch.randperm(subset.shape[0])][0:max_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

            subset = torch.cat([subset[subset == index], subset[subset != index]])
            sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)
            x = data.x[subset]
            
            protected_idx = torch.tensor([0], dtype=torch.long)
            sub_g = Data(
                x=x, 
                edge_index=sub_edge_index, 
                y=torch.tensor([current_label], dtype=torch.long), 
                index=torch.tensor([index], dtype=torch.long), 
                hop=torch.tensor([hop], dtype=torch.long),
                protected_idx=protected_idx,
            )

            all_subgraphs.append(sub_g)

        end_idx = len(all_subgraphs)
        node_to_subgraph_range.append((start_idx, end_idx))

    return all_subgraphs, node_to_subgraph_range

def negative_sampling(edge_index, num_nodes, num_samples, is_undirected=True, device='cpu'):
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), 
                                  size=(num_nodes, num_nodes), device=device)

    if is_undirected:
        adj = adj.coalesce()
        row, col = adj.indices()
        undirected_edge_index = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
        reversed_index = torch.cat([col.unsqueeze(0), row.unsqueeze(0)], dim=0)
        all_existing = torch.cat([undirected_edge_index, reversed_index], dim=1)
        existing_set = set((u.item(), v.item()) for u, v in all_existing.t())
    else:
        existing_set = set((u.item(), v.item()) for u, v in edge_index.t())

    total = num_nodes * num_nodes
    candidate_indices = torch.randint(0, total, (num_samples * 2,), device=device)
    us = candidate_indices // num_nodes
    vs = candidate_indices % num_nodes

    mask = (us != vs)
    us = us[mask]
    vs = vs[mask]

    candidate_pairs = torch.stack([us, vs], dim=1).tolist()
    neg_edges = []
    seen = set()

    for u, v in candidate_pairs:
        if (u, v) in existing_set or (u, v) in seen:
            continue
        neg_edges.append((u, v))
        seen.add((u, v))
        if is_undirected:
            seen.add((v, u))
        if len(neg_edges) >= num_samples:
            break

    neg_edges = torch.tensor(neg_edges, dtype=torch.long, device=device).t()
    return neg_edges

def positive_sampling(edge_index, num_samples, is_undirected=True):
    src, dst = edge_index
    mask = src != dst
    src, dst = src[mask], dst[mask]

    if is_undirected:
        keep = src < dst
        src, dst = src[keep], dst[keep]

    candidates = torch.stack([src, dst], dim=1)
    perm = torch.randperm(candidates.shape[0])[:num_samples]
    sampled = candidates[perm].t()
    return sampled


def induced_graphs_multi_hop_by_edge(data, k_hop=5):
    num_pos_samples = data.x.size(0)
    neg_ratio = 0.05
    num_neg_samples = int(num_pos_samples * neg_ratio)

    pos_edge_index = positive_sampling(data.edge_index, num_pos_samples, is_undirected=True)
    neg_edge_index = negative_sampling(data.edge_index, data.num_nodes, num_neg_samples, is_undirected=True, device=data.edge_index.device)

    sampled_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat([
        torch.ones(pos_edge_index.size(1), dtype=torch.long),
        torch.zeros(neg_edge_index.size(1), dtype=torch.long)
    ])

    perm = torch.randperm(sampled_edges.size(1))
    sampled_edges = sampled_edges[:, perm]
    labels = labels[perm]
    
    hop_range = (1, k_hop)
    all_subgraphs = []
    edge_to_subgraph_range = []

    edge_index = data.edge_index
    num_edges = sampled_edges.size(1)
    for edge_idx in tqdm(range(num_edges), desc="Generating edge-based subgraphs", unit="edge"):
        src = sampled_edges[0, edge_idx].item()
        dst = sampled_edges[1, edge_idx].item()
        edge_nodes = torch.tensor([src, dst], dtype=torch.long)
        current_label = labels[edge_idx].item()
        start_idx = len(all_subgraphs)
        src_label = data.y[src].item()
        dst_label = data.y[dst].item()
        src_candidates = torch.argwhere(data.y == src_label).flatten()
        dst_candidates = torch.argwhere(data.y == dst_label).flatten()
        
        for hop in range(hop_range[0], hop_range[1] + 1):
            min_size, max_size = hop * (hop + 1), (hop + 1) * (hop + 2)

            subset, _, _, _ = k_hop_subgraph(node_idx=edge_nodes, num_hops=hop, edge_index=edge_index)

            if len(subset) > max_size:
                subset = subset[torch.randperm(subset.shape[0])][:max_size - 1]
                subset = torch.unique(torch.cat([edge_nodes, subset]))

            subset = torch.cat([edge_nodes, subset[~torch.isin(subset, edge_nodes)]])
            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
            x = data.x[subset]

            sub_g = Data(
                x=x,
                edge_index=sub_edge_index,
                y=torch.tensor([current_label], dtype=torch.long), 
                edge=torch.tensor([[src], [dst]], dtype=torch.long),
                hop=torch.tensor([hop], dtype=torch.long),
                protected_idx=torch.tensor([0, 1], dtype=torch.long),
            )

            all_subgraphs.append(sub_g)

        end_idx = len(all_subgraphs)
        edge_to_subgraph_range.append((start_idx, end_idx))

    return all_subgraphs, edge_to_subgraph_range


def induced_graphs_multi_hop_by_given_edges(
    data: Data,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    k_hop: int = 5,
) -> Tuple[list, list]:
    hop_range = (1, k_hop)
    all_subgraphs = []
    edge_to_subgraph_range = []
    total_edges = pos_edges.size(1) + neg_edges.size(1)
    pbar = tqdm(total=total_edges, desc="building edge subgraphs", leave=False)

    def _process_edges(edge_tensor, label):
        for edge_idx in range(edge_tensor.size(1)):
            src = edge_tensor[0, edge_idx].item()
            dst = edge_tensor[1, edge_idx].item()
            edge_nodes = torch.tensor([src, dst], dtype=torch.long)
            current_label = label
            start_idx = len(all_subgraphs)

            for hop in range(hop_range[0], hop_range[1] + 1):
                min_size, max_size = hop * (hop + 1), (hop + 1) * (hop + 2)
                subset, _, _, _ = k_hop_subgraph(node_idx=edge_nodes, num_hops=hop, edge_index=data.edge_index)

                if len(subset) > max_size:
                    subset = subset[torch.randperm(subset.shape[0])][:max_size - 2]
                    subset = torch.unique(torch.cat([edge_nodes, subset]))

                subset = torch.cat([edge_nodes, subset[~torch.isin(subset, edge_nodes)]])
                sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

                sub_g = Data(
                    global_n_id=subset,
                    edge_index=sub_edge_index,
                    y=torch.tensor([current_label], dtype=torch.long),
                    edge=torch.tensor([[src], [dst]], dtype=torch.long),
                    hop=torch.tensor([hop], dtype=torch.long),
                    protected_idx=torch.tensor([0, 1], dtype=torch.long),
                )

                all_subgraphs.append(sub_g)

            end_idx = len(all_subgraphs)
            edge_to_subgraph_range.append((start_idx, end_idx))
            pbar.update(1)

    _process_edges(pos_edges, 1)
    _process_edges(neg_edges, 0)
    pbar.close()

    return all_subgraphs, edge_to_subgraph_range
