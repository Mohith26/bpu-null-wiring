# bpu-null-wiring

Does the fly's actual wiring matter in a Biological Processing Unit?

[Yu et al. 2025](https://arxiv.org/abs/2507.10951) turn the larval *Drosophila*
connectome (Winding et al. 2023, 2,952 neurons, 63,545 axon-to-dendrite edges)
into a fixed recurrent network: MNIST pixels are linearly projected into the
sensory neurons, activity runs through the frozen connectome for a few steps,
and a linear readout decodes the output neurons. They report 98% on MNIST
versus 97% for a parameter-matched MLP with a frozen random middle layer.

The paper does not compare against random graphs with the same statistics, so
it cannot tell "this specific wiring helps" apart from "any sparse recurrent
reservoir of this size helps". This repo runs that comparison under one
identical training pipeline.

## What I ran

Every condition uses the same 2,952 neurons, the same 430 sensory input
neurons, the same 218 output neurons (DN-SEZ + RGN), the same per-neuron
excitatory/inhibitory signs, the same trainable parameters (339,740), 4
recurrent steps, Adam (lr 1e-3, batch 128), and the recurrent matrix scaled to
spectral radius 5 separately for each graph.

| graph | what is kept | what is destroyed |
|---|---|---|
| connectome | everything | nothing |
| block_preserving | every neuron's in/out degree and in-strength, and the number of edges between every pair of cell types (sensory to PN, PN to LHN, ... to DN) | which specific neurons connect |
| degree_preserving | every neuron's in/out degree and in-strength | cell-type structure and specific wiring |
| erdos_renyi | number of edges and the multiset of synapse counts | everything else |
| frozen_mlp | (paper's baseline) same trainable parameter count | no graph at all |

Nulls are built by directed double-edge swaps (10 per edge, no self loops or
duplicate edges). 3 seeds per condition; each seed draws a new null graph,
new initial weights and, for the small regime, a new training subset.

## Results (MNIST test accuracy, %, mean and sd over 3 seeds)

| graph | full data, 5 epochs | 2,000 training images, 30 epochs | full data, unsigned weights |
|---|---:|---:|---:|
| connectome | 96.96 (0.07) | 91.72 (0.21) | 96.89 (0.35) |
| block_preserving | 97.06 (0.38) | 91.62 (0.31) | 96.80 (0.23) |
| degree_preserving | 97.03 (0.21) | 91.62 (0.28) | 96.31 (0.33) |
| erdos_renyi | 97.13 (0.25) | 92.29 (0.12) | 97.31 (0.18) |
| frozen_mlp | 97.58 (0.16) | 91.87 (0.13) | not run |

- No null does worse than the real connectome by more than 0.6 points, and the
  fully random Erdos-Renyi graph is the best reservoir in all three regimes
  (+0.17, +0.57 and +0.42 points). On MNIST, in this setup, the specific fly
  wiring carries no measurable advantage over random graphs with matched
  statistics.
- The frozen-MLP baseline matches the paper's 97%, but my connectome BPU
  reaches 97.0%, not 98%, so I do not reproduce the BPU's edge over the MLP.
- One plausible reason random graphs do slightly better: they route input to
  the readout faster. Within one hop of the sensory neurons, 95.7% of output
  neurons are reachable in Erdos-Renyi graphs versus 57.8% in the connectome
  (`results/reachability.csv`).

## Differences from the paper, stated plainly

- **Signs.** The paper signs each neuron from neurotransmitter annotations.
  The public Supplementary Data S1 has no neurotransmitter labels, so I use a
  seeded random Dale's-law assignment (20% inhibitory), identical across all
  graphs, plus an all-excitatory run. With real signs the absolute numbers
  could move; the comparison between graphs uses the same signs either way.
- **Scale and steps.** The paper does not give its weight normalisation. I
  chose spectral radius 5 from {0.9, 2, 5, 10, 20} using 1-epoch accuracy of
  the real connectome only, then froze it for every graph, which if anything
  favours the connectome. 4 steps covers all sensory-to-output paths (every
  output neuron is within 3 hops).
- **Scope.** MNIST only. I did not run CIFAR-10, ChessBench, the DCSBM
  expansions or the modality ablations. MNIST is easy enough that any decent
  reservoir may saturate; a harder task could separate the graphs.
- Single CPU (Apple Silicon arm64), Python 3.12, torch 2.15, 3 seeds. Each
  full-data run takes about 35 seconds.

## Run it

```
pip install -r requirements.txt
python src/fetch_data.py          # Winding et al. 2023 Data S1 (public mirror)
python src/run.py full            # also: small, unsigned (resumable)
python src/summarize.py           # results/RESULTS.md, summary.csv
pytest
```

`tests/` (10 tests) checks that each null keeps exactly what the table above
says it keeps (degrees, in-strength, cell-type-pair edge counts, edge count,
weight multiset, no self loops or duplicates), the matrix orientation and
spectral scaling, that activity only moves along edges one hop per step, that
the MLP baseline has exactly the BPU's trainable parameter count, and the
loaded connectome's size and pools.

Data: Winding, Pedigo et al., "The connectome of an insect brain", Science
2023, Supplementary Data S1. Please cite the original paper.
