import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .expert import Encoder


class Experts(nn.Module):
    def __init__(self, init_curvs, in_dim, hidden_dim, out_dim, learnable=True):
        super(Experts, self).__init__()
        self.experts = nn.ModuleList()
        num_factors = len(init_curvs)

        for curv in init_curvs:
            if curv == 0:
                self.experts.append(Encoder(0, in_dim, hidden_dim, out_dim, learnable=False))
            else:
                self.experts.append(Encoder(curv, in_dim, hidden_dim, out_dim, learnable))

        self.norm1 = nn.LayerNorm(num_factors * out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        embeds = []
        for expert in self.experts:
            embed = expert(x, edge_index)
            embeds.append(embed)
        embeds = torch.concat(embeds, -1)
        return self.norm1(embeds)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, topk_idx: torch.Tensor):
        num_subgraphs = x.size(0)
        num_experts = len(self.experts)
        out_dim = 32

        all_expert_outputs = torch.zeros(num_subgraphs, num_experts, out_dim,
                                         device=x.device, dtype=x.dtype)

        from torch_geometric.utils import add_self_loops
        edge_index_with_loops, _ = add_self_loops(edge_index)

        for expert_id in range(num_experts):
            mask = (topk_idx == expert_id).any(dim=1)
            selected_indices = mask.nonzero(as_tuple=True)[0]

            if len(selected_indices) == 0:
                continue

            sub_x = x[selected_indices]

            edge_mask = mask[edge_index_with_loops[0]] & mask[edge_index_with_loops[1]]
            sub_edge_index = edge_index_with_loops[:, edge_mask]

            node_mapping = torch.full((num_subgraphs,), -1, dtype=torch.long, device=x.device)
            node_mapping[selected_indices] = torch.arange(len(selected_indices), device=x.device)
            sub_edge_index = node_mapping[sub_edge_index]

            sub_output = self.experts[expert_id].encode_with_preprocessed_edges(sub_x, sub_edge_index)

            all_expert_outputs[selected_indices, expert_id] = sub_output

        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, all_expert_outputs.size(-1))
        selected_outputs = torch.gather(all_expert_outputs, 1, topk_idx_expanded)
        result = selected_outputs.reshape(num_subgraphs, -1)
        return result


class Gating(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int,
                 noisy_gating=False, train_temperature=2.0, eval_temperature=1.0):
        super(Gating, self).__init__()
        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, out_dim)
        self.pooling = global_mean_pool
        self.classifier = nn.Linear(out_dim, num_experts, bias=True)
        self.num_experts = num_experts
        self.train_temperature = train_temperature
        self.eval_temperature = eval_temperature

        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, sub_x, sub_edge_index):
        x = self.encoder1(sub_x, sub_edge_index)
        x = self.encoder2(x, sub_edge_index)

        logits = self.classifier(x)

        temperature = self.train_temperature if self.training else self.eval_temperature

        weights = F.softmax(logits / temperature, dim=-1)

        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balance_loss(weights)

        return weights, aux_loss

    def _compute_load_balance_loss(self, weights):
        expert_usage = weights.mean(dim=0)
        uniform = torch.ones_like(expert_usage) / self.num_experts
        loss = F.mse_loss(expert_usage, uniform)

        return loss


class GOGMoE(nn.Module):
    def __init__(self, emb_dim, num_experts=5, device='cuda', out_dim=0, topk=1, init_curvs=None):
        super().__init__()
        self.device = device

        all_curvs = init_curvs if init_curvs is not None else [-3, -1, 0, 1, 3]
        selected_curvs = all_curvs[:num_experts]
        self.num_experts = len(selected_curvs)
        self.topk = min(topk, self.num_experts)

        self.experts = Experts(
            init_curvs=selected_curvs,
            in_dim=emb_dim,
            hidden_dim=64,
            out_dim=32
        ).to(device)

        self.gating = Gating(
            in_dim=emb_dim,
            hidden_dim=64,
            out_dim=64,
            num_experts=self.num_experts,
        ).to(device)

        self.node_classifier = nn.Linear(self.topk * 32, out_dim).to(device)

    def forward(self, x, edge_index_list, k_hop=6, topm=None):
        num_subgraphs = x.size(0)
        num_centers = num_subgraphs // k_hop
        topm = self.topk if topm is None else min(max(1, topm), self.topk)

        expert_weights, aux_loss = self.gating(x, edge_index_list)

        center_expert_weights = expert_weights.reshape(num_centers, k_hop, -1).mean(dim=1)
        center_topk_weight, center_topk_idx = torch.topk(center_expert_weights, k=self.topk, dim=1)
        center_topk_weight = F.softmax(center_topk_weight, dim=1)

        overall_confidence = center_expert_weights.max(dim=1)[0].mean().detach()

        mask = (torch.arange(self.topk, device=center_topk_weight.device) < topm).float().unsqueeze(0)
        center_topk_weight = center_topk_weight * mask
        center_topk_weight = center_topk_weight / center_topk_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

        topk_idx = center_topk_idx.repeat_interleave(k_hop, dim=0)
        topk_weight = center_topk_weight.repeat_interleave(k_hop, dim=0)

        embedding = self.experts.encode(x, edge_index_list, topk_idx)

        experts_weight = topk_weight.repeat_interleave(32, dim=1)
        embedding = embedding * experts_weight

        out = self.node_classifier(embedding)

        final_output = out.reshape(num_centers, k_hop * out.size(-1))

        return [final_output], aux_loss, overall_confidence


__all__ = ["Experts", "Gating", "GOGMoE"]
