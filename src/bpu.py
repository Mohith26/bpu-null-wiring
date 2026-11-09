"""BPU-style reservoir: a fixed connectome as the recurrent core, trainable
input projection into the sensory neurons and linear readout from the output
neurons (DN-SEZ + RGN), following Yu et al. 2025 (arXiv 2507.10951)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from connectome import Connectome


def recurrent_matrix(c: Connectome, signs: np.ndarray, radius: float = 0.9) -> torch.Tensor:
    """Sparse (n x n) matrix W with W[post, pre] = sign[pre] * weight, scaled
    so its spectral radius is `radius` (each graph is scaled on its own, so
    nulls are compared at the same dynamical regime)."""
    vals = signs[c.pre] * c.weight
    W = torch.sparse_coo_tensor(
        np.vstack([c.post, c.pre]), torch.tensor(vals, dtype=torch.float32), (c.n, c.n)
    ).coalesce()
    rho = spectral_radius(W)
    return (W * (radius / rho)).coalesce() if rho > 0 else W


def spectral_radius(W: torch.Tensor) -> float:
    """Largest |eigenvalue| via dense eigvals (n is only ~3k)."""
    return float(np.max(np.abs(np.linalg.eigvals(W.to_dense().numpy()))))


class BPU(nn.Module):
    def __init__(
        self,
        W: torch.Tensor,
        sensory: np.ndarray,
        outputs: np.ndarray,
        d_in: int,
        n_cls: int,
        steps: int,
    ):
        super().__init__()
        self.register_buffer("W", W)
        self.register_buffer("sensory", torch.as_tensor(sensory))
        self.register_buffer("outputs", torch.as_tensor(outputs))
        self.n = W.shape[0]
        self.steps = steps
        self.inp = nn.Linear(d_in, len(sensory))
        self.out = nn.Linear(len(outputs), n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        drive = torch.zeros(B, self.n, device=x.device)
        drive[:, self.sensory] = self.inp(x)
        h = torch.zeros(B, self.n, device=x.device)
        for _ in range(self.steps):
            h = torch.relu(torch.sparse.mm(self.W, h.T).T + drive)
        return self.out(h[:, self.outputs])


class FrozenMLP(nn.Module):
    """The paper's baseline: trainable in -> h1, fixed random h1 -> h2,
    trainable h2 -> out, with h1/h2 widths matching the BPU's sensory and
    output pool sizes so trainable parameter counts match exactly."""

    def __init__(self, d_in: int, h1: int, h2: int, n_cls: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.inp = nn.Linear(d_in, h1)
        self.register_buffer("mid", torch.randn(h1, h2, generator=g) / np.sqrt(h1))
        self.out = nn.Linear(h2, n_cls)

    def forward(self, x):
        return self.out(torch.relu(torch.relu(self.inp(x)) @ self.mid))


def n_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
