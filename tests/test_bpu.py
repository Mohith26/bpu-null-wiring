import gzip
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bpu
import connectome as C
import mnist

warnings.filterwarnings("ignore", message="Sparse invariant checks")


def toy(n=60, m=400, seed=0):
    rng = np.random.default_rng(seed)
    flat = rng.choice([v for v in range(n * n) if v // n != v % n], size=m, replace=False)
    types = np.array(["sensory"] * 10 + ["PN"] * 20 + ["LHN"] * 20 + ["DN-SEZ"] * 6 + ["RGN"] * 4)
    return C.Connectome(np.arange(n), types, flat // n, flat % n, rng.integers(1, 9, m).astype(float))


def _simple(c):
    keys = c.pre * c.n + c.post
    assert len(np.unique(keys)) == len(keys), "duplicate edge"
    assert not np.any(c.pre == c.post), "self loop"


def _deg(c):
    return np.bincount(c.pre, minlength=c.n), np.bincount(c.post, minlength=c.n)


def test_degree_preserving_keeps_degrees_and_in_strength():
    c = toy()
    g = C.degree_preserving(c, np.random.default_rng(1))
    _simple(g)
    assert all(np.array_equal(a, b) for a, b in zip(_deg(c), _deg(g)))
    np.testing.assert_allclose(np.bincount(c.post, c.weight, c.n), np.bincount(g.post, g.weight, g.n))
    # and it actually rewired
    assert len(set(zip(c.pre, c.post)) & set(zip(g.pre, g.post))) < 0.5 * len(c.pre)


def test_block_preserving_keeps_type_to_type_counts_and_degrees():
    c = toy(seed=3)
    g = C.block_preserving(c, np.random.default_rng(2))
    _simple(g)
    assert all(np.array_equal(a, b) for a, b in zip(_deg(c), _deg(g)))

    def blocks(x):
        pairs = list(zip(x.celltype[x.pre], x.celltype[x.post]))
        return {p: pairs.count(p) for p in set(pairs)}

    assert blocks(c) == blocks(g)
    assert len(set(zip(c.pre, c.post)) & set(zip(g.pre, g.post))) < len(c.pre)


def test_erdos_renyi_keeps_edge_count_and_weight_multiset():
    c = toy()
    g = C.erdos_renyi(c, np.random.default_rng(4))
    _simple(g)
    assert len(g.pre) == len(c.pre)
    assert sorted(g.weight) == sorted(c.weight)


def test_dale_signs_fraction():
    s = C.dale_signs(10_000, 0.2, np.random.default_rng(0))
    assert set(np.unique(s)) == {-1.0, 1.0}
    assert abs((s < 0).mean() - 0.2) < 0.02


def test_recurrent_matrix_orientation_sign_and_radius():
    c = C.Connectome(
        np.arange(3),
        np.array(["sensory", "PN", "DN-SEZ"]),
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([2.0, 3.0]),
    )
    W = bpu.recurrent_matrix(c, np.array([1.0, -1.0, 1.0]), radius=1.0).to_dense()
    # nilpotent chain: spectral radius 0, so no rescaling; W[post, pre]
    assert W[1, 0] == 2.0 and W[2, 1] == -3.0 and W[0, 1] == 0
    c2 = toy()
    W2 = bpu.recurrent_matrix(c2, np.ones(c2.n), radius=0.7)
    assert bpu.spectral_radius(W2) == pytest.approx(0.7, rel=1e-4)


def test_bpu_propagates_only_along_edges():
    # sensory 0 -> 1 -> output 2; with 1 step the output sees nothing, with 2 it does
    c = C.Connectome(
        np.arange(3),
        np.array(["sensory", "PN", "DN-SEZ"]),
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([1.0, 1.0]),
    )
    W = bpu.recurrent_matrix(c, np.ones(3), radius=1.0)
    for steps, expect_zero in [(1, True), (2, True), (3, False)]:
        m = bpu.BPU(W, c.sensory, c.outputs, 1, 1, steps)
        with torch.no_grad():
            m.inp.weight.fill_(1.0)
            m.inp.bias.zero_()
            m.out.weight.fill_(1.0)
            m.out.bias.zero_()
            y = m(torch.ones(1, 1))
        assert (y.item() == 0) == expect_zero


def test_frozen_mlp_matches_bpu_trainable_count():
    c = toy()
    W = bpu.recurrent_matrix(c, np.ones(c.n))
    b = bpu.BPU(W, c.sensory, c.outputs, 784, 10, 4)
    f = bpu.FrozenMLP(784, len(c.sensory), len(c.outputs), 10, seed=0)
    assert bpu.n_trainable(b) == bpu.n_trainable(f)
    assert f(torch.zeros(2, 784)).shape == (2, 10)


def test_hops_to_outputs():
    c = C.Connectome(
        np.arange(4),
        np.array(["sensory", "PN", "DN-SEZ", "RGN"]),
        np.array([0, 1, 0]),
        np.array([1, 2, 3]),
        np.ones(3),
    )
    np.testing.assert_allclose(C.hops_to_outputs(c, 3), [0.5, 1.0, 1.0])


def test_parse_idx_roundtrip():
    arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    raw = (0x0803).to_bytes(4, "big") + b"".join(d.to_bytes(4, "big") for d in arr.shape) + arr.tobytes()
    np.testing.assert_array_equal(mnist.parse_idx(gzip.decompress(gzip.compress(raw))), arr)


@pytest.mark.skipif(not C.DATA.exists(), reason="connectome data not downloaded")
def test_real_connectome_shape_and_pools():
    c = C.load()
    assert c.n == 2952 and len(c.pre) == 63545
    assert len(c.sensory) == 430 and len(c.outputs) == 218
    # rows are presynaptic: sensory neurons receive no axon->dendrite input
    assert not np.isin(c.post, c.sensory).any()
