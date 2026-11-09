"""Train BPU reservoirs on MNIST with the real connectome and matched nulls.

Each run writes results/runs/<tag>.json and is skipped if that file exists,
so the sweep can be resumed in chunks:  python src/run.py <regime> [max_runs]
regimes: full (60k train, 5 epochs), small (2k train, 30 epochs),
unsigned (full regime, all-excitatory weights).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bpu
import connectome as C
import mnist

warnings.filterwarnings("ignore", message="Sparse invariant checks")
OUT = Path(__file__).resolve().parents[1] / "results" / "runs"
GRAPHS = {
    "connectome": lambda c, rng: c,
    "block_preserving": C.block_preserving,
    "degree_preserving": C.degree_preserving,
    "erdos_renyi": C.erdos_renyi,
}
REGIMES = {
    "full": dict(n_train=60000, epochs=5, signed=True),
    "small": dict(n_train=2000, epochs=30, signed=True),
    "unsigned": dict(n_train=60000, epochs=5, signed=False),
}
# RADIUS was picked from {0.9, 2, 5, 10, 20} by 1-epoch accuracy of the REAL
# connectome only, then frozen for every graph (if anything this favours it).
STEPS, FRAC_INH, BATCH, LR, RADIUS = 4, 0.2, 128, 1e-3, 5.0


def train_eval(model, tx, ty, sx, sy, epochs, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    X, Y = torch.from_numpy(tx), torch.from_numpy(ty)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for k in range(0, len(X), BATCH):
            idx = perm[k : k + BATCH]
            loss = torch.nn.functional.cross_entropy(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = torch.cat(
            [model(torch.from_numpy(sx[k : k + 1000])).argmax(1) for k in range(0, len(sx), 1000)]
        )
    return float((pred.numpy() == sy).mean())


def main(regime: str, max_runs: int = 1000, seeds=(0, 1, 2)) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = REGIMES[regime]
    tx, ty, sx, sy = mnist.load()
    base = C.load()
    done = 0
    conds = list(GRAPHS) + (["frozen_mlp"] if cfg["signed"] else [])
    for seed in seeds:
        sub = np.random.default_rng(1000 + seed).permutation(len(tx))[: cfg["n_train"]]
        for cond in conds:
            tag = f"{regime}__{cond}__s{seed}"
            path = OUT / f"{tag}.json"
            if path.exists() or done >= max_runs:
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            if cond == "frozen_mlp":
                model = bpu.FrozenMLP(784, len(base.sensory), len(base.outputs), 10, seed)
            else:
                graph = GRAPHS[cond](base, np.random.default_rng(seed))
                signs = (
                    C.dale_signs(base.n, FRAC_INH, np.random.default_rng(10_000 + seed))
                    if cfg["signed"]
                    else np.ones(base.n)
                )
                W = bpu.recurrent_matrix(graph, signs, RADIUS)
                model = bpu.BPU(W, base.sensory, base.outputs, 784, 10, STEPS)
            acc = train_eval(model, tx[sub], ty[sub], sx, sy, cfg["epochs"], seed)
            rec = dict(
                regime=regime,
                condition=cond,
                seed=seed,
                test_acc=acc,
                trainable=bpu.n_trainable(model),
                seconds=round(time.time() - t0, 1),
                **cfg,
            )
            path.write_text(json.dumps(rec))
            print(tag, round(acc, 4), rec["seconds"], "s", flush=True)
            done += 1


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
