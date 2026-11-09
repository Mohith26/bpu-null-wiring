# Results

## full

| condition         |   mean |   sd |   n |   diff_vs_connectome |
|:------------------|-------:|-----:|----:|---------------------:|
| connectome        |  96.96 | 0.07 |   3 |                 0.00 |
| block_preserving  |  97.06 | 0.38 |   3 |                 0.10 |
| degree_preserving |  97.03 | 0.21 |   3 |                 0.07 |
| erdos_renyi       |  97.13 | 0.25 |   3 |                 0.17 |
| frozen_mlp        |  97.58 | 0.16 |   3 |                 0.62 |

## small

| condition         |   mean |   sd |   n |   diff_vs_connectome |
|:------------------|-------:|-----:|----:|---------------------:|
| connectome        |  91.72 | 0.21 |   3 |                 0.00 |
| block_preserving  |  91.62 | 0.31 |   3 |                -0.09 |
| degree_preserving |  91.62 | 0.28 |   3 |                -0.09 |
| erdos_renyi       |  92.29 | 0.12 |   3 |                 0.57 |
| frozen_mlp        |  91.87 | 0.13 |   3 |                 0.16 |

## unsigned

| condition         |   mean |   sd |   n |   diff_vs_connectome |
|:------------------|-------:|-----:|----:|---------------------:|
| connectome        |  96.89 | 0.35 |   3 |                 0.00 |
| block_preserving  |  96.80 | 0.23 |   3 |                -0.09 |
| degree_preserving |  96.31 | 0.33 |   3 |                -0.59 |
| erdos_renyi       |  97.31 | 0.18 |   3 |                 0.42 |

## Output neurons reachable from sensory neurons (%), mean of 3 graph seeds

|                   |   <= 1 hops |   <= 2 hops |   <= 3 hops |   <= 4 hops |
|:------------------|------------:|------------:|------------:|------------:|
| connectome        |        57.8 |        85.3 |       100.0 |       100.0 |
| block_preserving  |        88.5 |        97.1 |       100.0 |       100.0 |
| degree_preserving |        58.3 |        99.4 |       100.0 |       100.0 |
| erdos_renyi       |        95.7 |       100.0 |       100.0 |       100.0 |
