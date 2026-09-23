# Computational Track Submission — Placement & Routing Strategy

## Overview

`solve()` combines a multi-strategy weighted placement with local search
refinement, and a SABRE-style look-ahead router with decay and
bidirectional refinement. Both are tested against six baseline benchmarks
and five freshly-generated random programs never used during development,
to check for overfitting, and against deliberately broken hardware graphs
to check for robustness.

## Placement Strategy

The baseline's identity placement (logical qubit `i` → physical qubit `i`)
ignores which qubits actually interact. Instead, `solve()` searches over
multiple placement strategies and keeps whichever scores best after
routing:

1. **Two orderings** of logical qubits are tried: busiest-first (ranked by
   total interaction count) and a BFS clustering of the interaction graph
   (so tightly-connected groups of qubits get placed together early).
   These give 2-opt local search two different starting points, which can
   land in different local optima.
2. For each ordering, every physical qubit is tried as the starting
   location for the first logical qubit in that ordering (20 seeds on
   this hardware graph), with the rest placed greedily by weighted
   proximity to already-placed qubits.
3. Each resulting placement is refined with **2-opt local search** — a
   standard Quadratic Assignment Problem heuristic: repeatedly try
   swapping which physical qubit two logical qubits occupy, keeping the
   swap only if it strictly reduces total weighted distance. This is
   mathematically guaranteed to terminate (cost strictly decreases over
   a finite set of placements) and can only ever improve or match the
   starting placement, never worsen it.
4. Each refined placement is further improved with one proper SABRE
   bidirectional pass (see Routing Strategy below).
5. The single best-scoring combination, across all orderings, seeds, and
   restarts, is kept.

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
  broken with a small, seeded random jitter, and each placement is tried
  3 times with different tie-breaks, keeping the best result. All
  randomness is seeded, so results are fully reproducible.
- **Post-processing**: any pair of adjacent SWAP operations on the exact
  same physical pair is removed — this is a mathematical identity (two
  identical SWAPs in a row cancel out exactly), not a heuristic, so it
  can only ever reduce SWAP count without affecting correctness.

A separate attempt to add explicit "depth awareness" (biasing SWAP choice
toward qubits idle longer, to target the score's `0.5×depth` term
directly) was tested and found to *increase* total score rather than
decrease it, and was discarded — a useful negative result confirming the
distance/decay-based heuristic above is closer to a local optimum for
this router design than that particular depth heuristic.

## Results

Final score comparison across the six provided benchmarks
(`score = swap_count + 0.5 × depth`, lower is better):

| Benchmark | Baseline | Ours | Improvement |
|---|---|---|---|
| ghz_star | 14.0 | 7.0 | -50.0% |
| chain_trotter | 15.0 | 4.5 | -70.0% |
| ladder_trotter | 35.5 | 6.5 | -81.7% |
| qaoa_random | 39.0 | 12.5 | -67.9% |
| dense_random | 122.0 | 46.0 | -62.3% |
| vqe_layers | 58.0 | 3.0 | -94.8% |
| **Total** | **283.5** | **79.5** | **-72.0%** |

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
| fresh_small | 32.0 | 7.0 | -78.1% |
| fresh_medium | 51.5 | 21.0 | -59.2% |
| fresh_dense | 156.0 | 58.0 | -62.8% |
| fresh_sparse | 52.0 | 5.5 | -89.4% |
| fresh_large | 262.0 | 93.0 | -64.5% |
| **Total** | **553.5** | **184.5** | **-66.7%** |

The gap between known-benchmark improvement (-72.0%) and fresh-benchmark
improvement (-66.7%) is 5.3 percentage points — small enough to conclude
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