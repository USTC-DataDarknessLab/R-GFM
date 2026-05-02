import torch
import torch.nn.functional as F


def graphcl_nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    assert z1.shape == z2.shape and z1.dim() == 2
    N = z1.size(0)

    Z = torch.cat([z1, z2], dim=0)
    Z = F.normalize(Z, dim=1)
    sim = (Z @ Z.t()) / tau

    pos = torch.arange(2 * N, device=Z.device)
    pos = (pos + N) % (2 * N)

    mask = torch.ones((2 * N, 2 * N), dtype=torch.bool, device=Z.device)
    mask.fill_diagonal_(False)

    exp_sim = torch.exp(sim) * mask
    num = torch.exp(sim[torch.arange(2 * N, device=Z.device), pos])
    den = exp_sim.sum(dim=1).clamp_min(1e-12)

    return (-torch.log(num / den)).mean()


__all__ = ["graphcl_nt_xent"]
