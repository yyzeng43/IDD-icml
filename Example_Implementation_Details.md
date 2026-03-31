**Table R4.** Example Implementation Details and Hyperparameters. CNF is the abbreviation of Conditional Normalizing Flows.
| | Synthetic ($d$=1) | Synthetic ($d$=5) | Synthetic ($d$=10) | Synthetic ($d$=50) | FlowCAP-II (AML) | Reddit Sentiment |
|---|---|---|---|---|---|---|
| **Data** | | | | | | |
| Dimension $d$ | 1 | 5 | 10 | 50 | 7 | 20 |
| Batch size $N_t$ | {50, 100, 300} | {50, 100, 300} | {50, 100, 300} | {50, 100, 300} | ~100–300 | ~50–200 |
| Pre-change $n_0$ | 300 | 300 | 300 | 300 | 300 | 50 |
| MFPCA truncation $K$ | CVE ≥ 0.95 | CVE ≥ 0.95 | CVE ≥ 0.95 | CVE ≥ 0.95 | CVE ≥ 0.95 | CVE ≥ 0.95 |
| **OT Solver** | | | | | | |
| Method | Exact LP | Exact LP | Sinkhorn | Sinkhorn | Sinkhorn | Sinkhorn |
| Library call | `ot.emd()` | `ot.emd()` | `ot.sinkhorn()` | `ot.sinkhorn()` | `ot.sinkhorn()` | `ot.sinkhorn()` |
| `reg` (ε) | — | — | 0.05 | 0.05 | 0.05 | 0.05 |
| `numIterMax` | — | — | 5000 | 5000 | 5000 | 5000 |
| `stopThr` | — | — | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `use_eps_scaling` | — | — | True | True | True | True |
| **Barycenter** | | | | | | |
| Method | Closed-form | Fixed-support Sinkhorn | CNF | CNF | CNF | CNF |
| `n_bary` | — | 512 | — | — | — | — |
| Sinkhorn inner iters | — | 500 | — | — | — | — |
| Fixed-point outer iters | — | 300 | — | — | — | — |
| `flow_bar_hidden` | — | — | 32 | 32 | 32 | 32 |
| `flow_bar_blocks` | — | — | 8 | 8 | 8 | 8 |
| `flow_ot_hidden` | — | — | 64 | 64 | 64 | 64 |
| `flow_ot_blocks` | — | — | 16 | 16 | 16 | 16 |
| `epochs` | — | — | 500 | 500 | 500 | 500 |
| `batch_size` (training) | — | — | 2048 | 2048 | 2048 | 2048 |
| `lr` | — | — | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| `grad_clip` | — | — | 2.0 | 2.0 | 2.0 | 2.0 |
| Temp schedule | — | — | 1.0 → 1e-2 | 1.0 → 1e-2 | 1.0 → 1e-2 | 1.0 → 1e-2 |
| LR scheduler | — | — | Plateau (0.8, pat=1000) | Plateau (0.8, pat=1000) | Plateau (0.8, pat=1000) | Plateau (0.8, pat=1000) |