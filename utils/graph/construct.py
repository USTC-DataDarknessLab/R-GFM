import torch
import torch.nn.functional as F


def build_center_similarity_edges(center_ranges, x_all, density=0.5, device='cuda'):
    if len(center_ranges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    k = center_ranges[0][1] - center_ranges[0][0]
    num_centers = len(center_ranges)
    num_edges_per_center = k * (k - 1) // 2
    num_sample = int(density * num_edges_per_center)

    if num_sample == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    x_batched = x_all.view(num_centers, k, -1)

    x_normed = F.normalize(x_batched + 1e-6, p=2, dim=-1)
    sim = torch.bmm(x_normed, x_normed.transpose(1, 2))

    triu_row, triu_col = torch.triu_indices(k, k, offset=1, device=x_all.device)

    sim_scores = sim[:, triu_row, triu_col]

    probs = F.softmax(sim_scores, dim=1)
    sampled_idx = torch.multinomial(probs, num_sample, replacement=False)

    src = triu_row[sampled_idx]
    dst = triu_col[sampled_idx]

    offsets = torch.tensor([l for l, r in center_ranges], device=x_all.device).unsqueeze(1)
    src_global = src + offsets
    dst_global = dst + offsets

    src_flat = src_global.flatten()
    dst_flat = dst_global.flatten()

    edge_index = torch.stack([
        torch.cat([src_flat, dst_flat]),
        torch.cat([dst_flat, src_flat])
    ], dim=0)

    return edge_index.to(device)
