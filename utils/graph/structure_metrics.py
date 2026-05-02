import math
from typing import Iterable, List, NamedTuple, Optional, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


class StructureStats(NamedTuple):
    mean_nodes: float
    std_nodes: float
    cv_nodes: float
    mean_edges: float
    std_edges: float
    cv_edges: float
    mean_density: float
    std_density: float
    cv_density: float
    deg_cv_mean: float
    deg_cv_std: float


def _safe_mean_std(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean())
    std = float(t.std(unbiased=False))
    return mean, std


def _safe_cv(mean: float, std: float) -> float:
    if mean == 0.0:
        return 0.0
    return std / mean


def _num_nodes(g: Data) -> int:
    if hasattr(g, "num_nodes") and g.num_nodes is not None:
        return int(g.num_nodes)
    if hasattr(g, "x") and g.x is not None:
        return int(g.x.size(0))
    raise ValueError("graph missing node information")


def _num_edges(g: Data) -> int:
    if not hasattr(g, "edge_index") or g.edge_index is None:
        raise ValueError("graph missing edge information")
    return int(g.edge_index.size(1))


def _degree_cv(g: Data) -> float:
    num_nodes = _num_nodes(g)
    if num_nodes == 0:
        return 0.0
    d = degree(g.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    mean = float(d.mean())
    std = float(d.std(unbiased=False))
    return _safe_cv(mean, std)


def compute_structure_stats(graphs: Iterable[Data]) -> StructureStats:
    nodes: List[float] = []
    edges: List[float] = []
    density: List[float] = []
    deg_cvs: List[float] = []

    for g in graphs:
        n = _num_nodes(g)
        e = _num_edges(g)
        nodes.append(float(n))
        edges.append(float(e))
        denom = max(float(n * (n - 1)), 1.0)
        density.append(min(float(e) / denom, 1.0))
        deg_cvs.append(_degree_cv(g))

    mean_nodes, std_nodes = _safe_mean_std(nodes)
    mean_edges, std_edges = _safe_mean_std(edges)
    mean_density, std_density = _safe_mean_std(density)

    mean_deg_cv, std_deg_cv = _safe_mean_std(deg_cvs)

    return StructureStats(
        mean_nodes=mean_nodes,
        std_nodes=std_nodes,
        cv_nodes=_safe_cv(mean_nodes, std_nodes),
        mean_edges=mean_edges,
        std_edges=std_edges,
        cv_edges=_safe_cv(mean_edges, std_edges),
        mean_density=mean_density,
        std_density=std_density,
        cv_density=_safe_cv(mean_density, std_density),
        deg_cv_mean=mean_deg_cv,
        deg_cv_std=std_deg_cv,
    )


def compute_diversity_score(
    stats: StructureStats,
    num_graphs: int = 1,
    deg_cv_cap: float = 2.0,
    spread_weight: float = 0.4,
    target_graphs: int = 11,
    scale_factor: float = 1.65,
) -> float:
    g = max(1, num_graphs)
    deg_mean = min(stats.deg_cv_mean, deg_cv_cap) / deg_cv_cap
    deg_std = min(stats.deg_cv_std, deg_cv_cap) / deg_cv_cap
    base = (deg_mean + spread_weight * deg_std) / (1.0 + spread_weight)

    dataset_factor = min(g / max(1, target_graphs), 1.0)
    score = base * dataset_factor * scale_factor
    return float(min(max(score, 0.0), 1.0))


def suggest_num_experts(
    diversity_score: float,
    n_N: int,
    A: float = 6.0,
    B: float = 3.0,
    min_experts: int = 1,
    max_experts: int = 5,
) -> int:
    if min_experts < 1 or max_experts < min_experts:
        raise ValueError(f"invalid expert bounds: min_experts={min_experts}, max_experts={max_experts}")

    if n_N <= 0:
        return min_experts

    S = float(min(max(diversity_score, 0.0), 1.0))

    min_risk = float('inf')
    best_j = min_experts

    for j in range(min_experts, max_experts + 1):
        diversity_term = (A * S) / j
        capacity_term = B * math.sqrt(j / n_N)
        R_j = diversity_term + capacity_term

        if R_j < min_risk:
            min_risk = R_j
            best_j = j

    return best_j


def compute_implied_zeta(diversity_score: float, num_experts: int) -> float:
    S = float(min(max(diversity_score, 0.0), 1.0))
    if S == 0.0:
        return float('inf')
    return num_experts / S
