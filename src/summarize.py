"""Aggregate results/runs/*.json into results/RESULTS.md and results/summary.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import connectome as C

ROOT = Path(__file__).resolve().parents[1] / "results"
ORDER = [
    "connectome",
    "block_preserving",
    "degree_preserving",
    "erdos_renyi",
    "frozen_mlp",
]


def main() -> None:
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "runs").glob("*.json"))]
    df = pd.DataFrame(rows)
    g = df.groupby(["regime", "condition"])["test_acc"]
    s = pd.DataFrame({"mean": g.mean() * 100, "sd": g.std(ddof=1) * 100, "n": g.size()}).reset_index()
    s["condition"] = pd.Categorical(s["condition"], ORDER, ordered=True)
    s = s.sort_values(["regime", "condition"])
    s.to_csv(ROOT / "summary.csv", index=False)

    lines = ["# Results\n"]
    for regime, part in s.groupby("regime", sort=False):
        ref = part.loc[part.condition == "connectome", "mean"].iloc[0]
        part = part.assign(diff_vs_connectome=part["mean"] - ref)
        lines.append(f"## {regime}\n")
        lines.append(part.drop(columns="regime").to_markdown(index=False, floatfmt=".2f") + "\n")

    base = C.load()
    hop = {"connectome": C.hops_to_outputs(base, 4)}
    for name, f in [
        ("block_preserving", C.block_preserving),
        ("degree_preserving", C.degree_preserving),
        ("erdos_renyi", C.erdos_renyi),
    ]:
        hop[name] = np.mean(
            [C.hops_to_outputs(f(base, np.random.default_rng(s_)), 4) for s_ in range(3)],
            axis=0,
        )
    h = pd.DataFrame(hop, index=[f"<= {k} hops" for k in range(1, 5)]).T * 100
    h.to_csv(ROOT / "reachability.csv")
    lines.append("## Output neurons reachable from sensory neurons (%), mean of 3 graph seeds\n")
    lines.append(h.to_markdown(floatfmt=".1f") + "\n")
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
