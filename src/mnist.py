"""MNIST from the public PyTorch mirror (idx files), cached under data/."""

from __future__ import annotations

import gzip
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def parse_idx(raw: bytes) -> np.ndarray:
    magic = int.from_bytes(raw[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


CACHE = Path(__file__).resolve().parents[1] / "data" / "mnist"


def load(cache: Path = CACHE):
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, f in FILES.items():
        p = cache / f
        if not p.exists():
            urlretrieve(URL + f, p)
        out[k] = parse_idx(gzip.decompress(p.read_bytes()))
    tx = out["train_x"].reshape(-1, 784).astype(np.float32) / 255.0
    sx = out["test_x"].reshape(-1, 784).astype(np.float32) / 255.0
    return tx, out["train_y"].astype(np.int64), sx, out["test_y"].astype(np.int64)
