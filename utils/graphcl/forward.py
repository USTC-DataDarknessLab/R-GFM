from typing import Sequence, Tuple

import torch
from torch import nn
from torch_geometric.utils import scatter

from utils.graph import build_center_similarity_edges
from utils.data import GraphBatch


def stage1_forward(
    encoder: nn.Module,
    projector: nn.Module,
    batch,
    center_ranges: Sequence[Tuple[int, int]] = None,
    use_sim_agg: bool = False,
    sim_alpha: float = 0.1,
    density: float = 0.6,
) -> torch.Tensor:
    batch = _normalize_batch(batch)
    g = encoder(batch.x, batch.edge_index, batch=batch.batch)

    if center_ranges is None:
        return projector(g)

    if use_sim_agg and len(center_ranges) > 0:
        edge_index = build_center_similarity_edges(
            center_ranges, g, device=g.device, density=density
        )
        row, col = edge_index
        msg = g[col]
        agg = scatter(msg, row, dim=0, dim_size=g.size(0), reduce='mean')
        g = g + sim_alpha * agg

    if len(center_ranges) > 0:
        num_centers = len(center_ranges)
        k_per_center = center_ranges[0][1] - center_ranges[0][0]

        all_same_k = all(r - l == k_per_center for l, r in center_ranges)
        if all_same_k and num_centers * k_per_center == g.size(0):
            center_features = g.view(num_centers, -1)
        else:
            center_features = []
            for l, r in center_ranges:
                center_features.append(g[l:r].flatten())
            center_features = torch.stack(center_features, dim=0)
    else:
        center_features = g.view(0, -1)

    return projector(center_features)


def stage2_forward(
    encoder: nn.Module,
    moe: nn.Module,
    projector: nn.Module,
    batch,
    center_ranges: Sequence[Tuple[int, int]],
    k_hop: int,
    density: float,
) -> torch.Tensor:
    batch = _normalize_batch(batch)
    g = encoder(batch.x, batch.edge_index, batch=batch.batch)
    edge_index = build_center_similarity_edges(center_ranges, g, device=g.device, density=density)
    moe_out_list = moe(g, edge_index, k_hop=k_hop)
    moe_out = torch.cat(moe_out_list, dim=0)
    z = projector(moe_out)
    return z


def _normalize_batch(batch: GraphBatch) -> GraphBatch:
    if batch.batch.numel() == 0:
        return batch
    unique, inv = torch.unique(batch.batch, sorted=True, return_inverse=True)
    if inv.max().item() == batch.batch.max().item():
        return batch
    return GraphBatch(batch.x, batch.edge_index, inv)


__all__ = ["stage1_forward", "stage2_forward"]
