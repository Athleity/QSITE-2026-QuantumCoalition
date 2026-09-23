# Computational Track Submission — Placement & Routing Strategy

## Overview

`solve()` combines two components: a weighted-centrality qubit placement
strategy, and a SABRE-style look-ahead router with decay and bidirectional
refinement. Both are tested against six baseline benchmarks and five
freshly-generated random programs never used during development, to check
for overfitting, and against deliberately broken hardware graphs to check
for robustness.

## Placement Strategy

The baseline's identity placement (logical qubit `i` → physical qubit `i`)
ignores which qubits actually interact. Instead:

1. Weight each pair of logical qubits by how many times they interact
   across the program.
2. Order logical qubits by total interaction weight (busiest first).
3. Try every physical qubit as the starting location for the busiest
   logical qubit (20 seeds on this hardware graph), then greedily place
   every remaining logical qubit as close as possible — weighted by
   interaction frequency — to qubits already placed.
4. Keep whichever seed produces the best final score after routing.

This alone (without any routing improvement) cut total score across the
six benchmarks from 283.5 to 205.0 (-27.7%).

## Routing Strategy

The baseline routes each gate independently via plain shortest-path SWAP
insertion, with no awareness of future gates. This was replaced with a
SABRE-style heuristic router (Li et al., "Tackling the Qubit Mapping
Problem for NISQ-Era Quantum Devices," ASPLOS 2019), which remains the
de facto industry standard and underlies Qiskit's current transpiler
(LightSABRE, Zou et al., arXiv:2409.08368):

- **Look-ahead cost function**: each candidate SWAP is scored not just by
  how much it helps the *current* blocked gate, but by a decayed sum of
  distances for the next several upcoming gates (window of 8, decay 0.5
  per step), so a SWAP that helps now but strands a qubit badly for the
  next gate is penalized.
- **Decay penalty**: qubits recently involved in a SWAP accrue a small
  cost penalty (reset periodically), discouraging the router from
  oscillating the same pair of qubits back and forth without progress.
- **Bidirectional refinement**: following the original SABRE paper's
  method exactly (one pass, not iterated), each candidate placement is
  refined by routing the program forward, then routing the *reversed*
  program starting from that final layout, then taking that result's
  final layout as the true initial placement for one last forward pass.
  An earlier attempt that repeated this loop three times performed worse
  than a single correct pass, confirming the paper's design choice.
- **Randomized restarts**: ties between near-equal candidate SWAPs are
  broken with a small, seeded random jitter, and each placement seed is
  tried 3 times with different tie-breaks, keeping the best result. All
  randomness is seeded, so results are fully reproducible.

## Results

Final score comparison across the six provided benchmarks
(`score = swap_count + 0.5 × depth`, lower is better):

| Benchmark | Baseline | Ours | Improvement |
|---|---|---|---|
| ghz_star | 14.0 | 7.0 | -50.0% |
| chain_trotter | 15.0 | 4.5 | -70.0% |
| ladder_trotter | 35.5 | 9.0 | -74.6% |
| qaoa_random | 39.0 | 14.5 | -62.8% |
| dense_random | 122.0 | 49.0 | -59.8% |
| vqe_layers | 58.0 | 3.0 | -94.8% |
| **Total** | **283.5** | **87.0** | **-69.3%** |

All results are valid under the scorer's full correctness checks (every
2Q gate on a real hardware edge, exact original program order preserved
after stripping SWAPs).

## Generalization Check

To rule out overfitting to the six known benchmarks, `solve()` was also
run against five freshly-generated random programs (varying sizes from
8–20 qubits and 12–60 gates, seeded independently, never used while
tuning the algorithm):

| Benchmark | Baseline | Ours | Improvement |
|---|---|---|---|
| fresh_small | 32.0 | 8.0 | -75.0% |
| fresh_medium | 51.5 | 22.5 | -56.3% |
| fresh_dense | 156.0 | 60.0 | -61.5% |
| fresh_sparse | 52.0 | 7.5 | -85.6% |
| fresh_large | 262.0 | 106.0 | -59.5% |
| **Total** | **553.5** | **204.0** | **-63.1%** |

The gap between known-benchmark improvement (-69.3%) and fresh-benchmark
improvement (-63.1%) is 6.2 percentage points — small enough to conclude
the algorithm generalizes rather than exploiting quirks of the six
provided benchmarks.

## Robustness

`solve()` explicitly handles two edge cases that would otherwise crash or
hang:

1. **Disconnected hardware graphs**: the algorithm restricts its search to
   a single connected region of the hardware graph large enough to hold
   the program, rather than assuming full connectivity.
2. **Unroutable programs**: if no connected region of the hardware graph
   has enough qubits for the program, `solve()` raises a clear
   `ValueError` explaining exactly how many qubits were needed versus
   available, rather than looping forever or producing an invalid
   result. This mirrors how a production quantum compiler behaves when a
   circuit genuinely cannot be mapped to given hardware — it is a
   mathematical impossibility (no SWAP sequence can move information
   across a hardware graph with no connecting edge), not an algorithmic
   shortfall.

## References

- Li, G., Ding, Y., & Xie, Y. (2019). *Tackling the Qubit Mapping Problem
  for NISQ-Era Quantum Devices*. ASPLOS 2019.
- Zou, H., Treinish, M., Hartman, K., Ivrii, A., & Lishman, J. (2024).
  *LightSABRE: A Lightweight and Enhanced SABRE Algorithm*.
  arXiv:2409.08368.