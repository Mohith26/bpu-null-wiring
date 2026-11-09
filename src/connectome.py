"""Larval Drosophila connectome (Winding et al. 2023) and matched null graphs.

All graphs are returned as a directed weighted edge list (pre, post, weight)
over the same 2,952 neurons, so every null keeps the neuron set, the sensory
and output pools, and the per-neuron sign used by the reservoir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "Supplementary-Data-S1"
OUTPUT_TYPES = ("DN-SEZ", "RGN")


@dataclass
class Connectome:
    ids: np.ndarray  # skeleton ids, length n
    celltype: np.ndarray  # str per neuron ("unannotated" if missing)
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray  # synapse counts (positive)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def sensory(self) -> np.ndarray:
        return np.flatnonzero(self.celltype == "sensory")

    @property
    def outputs(self) -> np.ndarray:
        return np.flatnonzero(np.isin(self.celltype, OUTPUT_TYPES))

    def with_edges(self, pre, post, weight) -> Connectome:
        return Connectome(
            self.ids,
            self.celltype,
            np.asarray(pre),
            np.asarray(post),
            np.asarray(weight),
        )


def load(data_dir: Path = DATA, matrix: str = "ad") -> Connectome:
    """Axon-to-dendrite matrix by default (rows presynaptic), as in the BPU paper."""
    A = pd.read_csv(data_dir / f"{matrix}_connectivity_matrix.csv", index_col=0)
    ids = A.index.to_numpy(dtype=np.int64)
    assert list(map(str, ids)) == list(A.columns)
    ann = pd.read_csv(data_dir / "annotations.csv", dtype=str)
    ct: dict[int, str] = {}
    for _, r in ann.iterrows():
        for side in (r.left_id, r.right_id):
            if side != "no pair":
                ct[int(side)] = r.celltype
    celltype = np.array([ct.get(int(i), "unannotated") for i in ids])
    M = A.to_numpy()
    pre, post = np.nonzero(M)
    return Connectome(ids, celltype, pre, post, M[pre, post].astype(np.float64))


# Null models -----------------------------------------------------------------


def erdos_renyi(c: Connectome, rng: np.random.Generator) -> Connectome:
    """Same number of edges placed uniformly (no self loops, no duplicates);
    the real weights are shuffled onto them."""
    n, m = c.n, len(c.pre)
    chosen: set[int] = set()
    while len(chosen) < m:
        k = rng.integers(0, n * n, size=2 * (m - len(chosen)))
        for v in k:
            if v // n != v % n:
                chosen.add(int(v))
            if len(chosen) == m:
                break
    flat = np.fromiter(chosen, dtype=np.int64)
    return c.with_edges(flat // n, flat % n, rng.permutation(c.weight))


def degree_preserving(c: Connectome, rng: np.random.Generator, swaps_per_edge: int = 10) -> Connectome:
    """Directed double-edge swaps (a->b, c->d) => (a->d, c->b), each new edge
    taking the weight of the edge whose target it inherits.

    Preserved exactly for every neuron: in-degree, out-degree and in-strength
    (total synapses received). Out-strength can drift because a source may
    end up holding a different weight. No self loops or duplicate edges."""
    pre = c.pre.copy()
    post = c.post.copy()
    w = c.weight.copy()
    m = len(pre)
    n = c.n
    edges = set((pre * n + post).tolist())
    target = swaps_per_edge * m
    done = 0
    while done < target:
        i, j = rng.integers(0, m, size=2)
        a, b, cc, d = pre[i], post[i], pre[j], post[j]
        if i == j or a == d or cc == b:
            continue
        e1, e2 = a * n + d, cc * n + b
        if e1 in edges or e2 in edges:
            continue
        edges.discard(a * n + b)
        edges.discard(cc * n + d)
        edges.add(e1)
        edges.add(e2)
        post[i], post[j] = d, b
        w[i], w[j] = w[j], w[i]
        done += 1
    return c.with_edges(pre, post, w)


def block_preserving(c: Connectome, rng: np.random.Generator, swaps_per_edge: int = 10) -> Connectome:
    """Degree-preserving swaps restricted to edge pairs whose targets share a
    cell type, so every (pre cell type -> post cell type) edge count is kept:
    the sensory -> projection -> ... -> output flow survives, the specific
    neuron-to-neuron wiring does not."""
    pre = c.pre.copy()
    post = c.post.copy()
    w = c.weight.copy()
    m, n = len(pre), c.n
    edges = set((pre * n + post).tolist())
    groups: dict[str, np.ndarray] = {}
    for t in np.unique(c.celltype):
        groups[t] = np.flatnonzero(c.celltype[post] == t)
    keys = [k for k, v in groups.items() if len(v) > 1]
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    probs = sizes / sizes.sum()
    target = swaps_per_edge * m
    done = tries = 0
    while done < target and tries < 50 * target:
        tries += 1
        g = groups[keys[rng.choice(len(keys), p=probs)]]
        i, j = g[rng.integers(0, len(g), size=2)]
        a, b, cc, d = pre[i], post[i], pre[j], post[j]
        if i == j or a == d or cc == b:
            continue
        e1, e2 = a * n + d, cc * n + b
        if e1 in edges or e2 in edges:
            continue
        edges.discard(a * n + b)
        edges.discard(cc * n + d)
        edges.add(e1)
        edges.add(e2)
        post[i], post[j] = d, b
        w[i], w[j] = w[j], w[i]
        done += 1
    return c.with_edges(pre, post, w)


def dale_signs(n: int, frac_inhibitory: float, rng: np.random.Generator) -> np.ndarray:
    s = np.ones(n)
    s[rng.random(n) < frac_inhibitory] = -1.0
    return s


def hops_to_outputs(c: Connectome, max_hops: int = 10) -> np.ndarray:
    """Fraction of output neurons reachable from any sensory neuron within h
    hops, h = 1..max_hops (unweighted BFS on the directed graph)."""
    adj = [[] for _ in range(c.n)]
    for a, b in zip(c.pre.tolist(), c.post.tolist()):
        adj[a].append(b)
    dist = np.full(c.n, np.inf)
    frontier = list(c.sensory)
    dist[frontier] = 0
    h = 0
    while frontier and h < max_hops:
        h += 1
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] == np.inf:
                    dist[v] = h
                    nxt.append(v)
        frontier = nxt
    out = dist[c.outputs]
    return np.array([(out <= k).mean() for k in range(1, max_hops + 1)])
