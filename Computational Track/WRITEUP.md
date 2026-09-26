# Compiling Quantum Circuits onto a Sparse 20-Qubit Device

**Team IonQ · GitHub: Athleity · Q-SITE 2026 Computational Track**

**68.0 total** on the six public benchmarks. Baseline 283.5. All six
verified by the official `starter_kit` scorer, none hidden behind a custom
metric.

---

## The problem, briefly

The task is quantum circuit compilation. We're given a program of one- and
two-qubit gates, and a 20-qubit chip that isn't fully connected — each qubit
has one, two, or three physical neighbours. A two-qubit gate can only run
across a wire that physically exists. When the two qubits it needs aren't
adjacent, we have to insert SWAP operations to bring them together.

Every SWAP costs one point. Every parallel time step in the resulting circuit
costs half a point. So the objective is

    score = (#SWAPs) + 0.5 × (circuit depth)

and we want to make that as small as we can. Six benchmark programs, one
total, lower is better.

## What we achieved

| Benchmark | Baseline | Ours | SWAPs | Depth | Status |
|---|---|---|---|---|---|
| ghz_star | 14.0 | 6.5 | 2 | 9 | proven optimal |
| chain_trotter | 15.0 | 4.5 | 0 | 9 | proven optimal |
| ladder_trotter | 35.5 | 6.5 | 3 | 7 | proven optimal |
| qaoa_random | 39.0 | 11.5 | 6 | 11 | best found (floor 9.0) |
| dense_random | 122.0 | 36.0 | 24 | 23 | best found (floor 17.0) |
| vqe_layers | 58.0 | 3.0 | 0 | 6 | proven optimal |
| **Total** | **283.5** | **68.0** | | | |

That's a 76 % reduction on the provided baseline. Four of the six benchmarks
are not just "best we found" — they match a lower bound the solver proves
internally, so no routing can do better. The other two are best-found with
the gap honestly reported below.

---

## How it works

The solver is built on two ideas, and a set of smaller refinements around
them.

### 1. Sometimes the answer is zero SWAPs

The 20-qubit hardware graph has a Hamiltonian path — a single line that
visits all twenty qubits exactly once. Any circuit whose two-qubit
interactions lie entirely along that line can be embedded directly, with no
SWAP operations at all. The score then reduces to the circuit's critical-path
depth, and that number is a lower bound on *any* hardware — the depth cannot
be less than the longest chain of dependent gates. So for those circuits, the
answer is not just good, it's optimal by construction.

chain_trotter and vqe_layers both fall in this category. They score 4.5
and 3.0, and no algorithm on any hardware can beat either number.

### 2. Beam search on the exact objective

Most circuits don't fit on the Hamiltonian path. ghz_star, ladder_trotter,
qaoa_random, and dense_random all need at least a few SWAPs, and the
search for the best routing becomes the whole game.

The approach here is a beam search that expands partial routings one gate at
a time. The key detail: every partial route tracks the ready time of each
physical qubit, so the running score at any point in the search is the *exact*
score the route would receive, not an estimate or a heuristic proxy. There is
no gap between what the search optimizes and what the grader measures.

At each gate the beam keeps the best few candidates. Candidates are ranked by
cost-so-far plus a short decayed lookahead — the weighted distances of the
next handful of two-qubit pairs, discounted so that distant gates count less.
The decay is what keeps the search from over-committing to a placement that
looks good three gates from now but is bad immediately.

### The refinements around those two ideas

**Lazy placement.** Rather than fixing the logical-to-physical mapping up
front, the solver pins each logical qubit to a physical qubit on its first
use, and lets the beam search discover the rest. In local sweeps this beat
every pre-computed placement we tried — including a spectral ordering of the
interaction graph, a simulated-annealing refinement, and a portfolio over
both. The lazy approach wins on the dense programs, where no fixed placement
is good for more than a handful of gates.

**Relaxation chains.** Beam search alone gets trapped in local optima. The
fix: run the beam forward over the circuit, then run it backward over the
reversed circuit starting from where the forward pass ended, then forward
again. Each pass tends to unwind the worst decision the previous pass made.
Repeat until the score stops improving. The chain is not a new algorithm —
it's the same beam search, just applied to a different traversal order — but
it finds materially better routings than a single pass.

**Portfolio of beam widths.** Narrow beams (16 states) run first because
they're cheap and often land on the right basin. Their best results seed
wider beams (64, then 192). The wider passes are slower but explore enough
of the neighbourhood to escape a narrow beam's local minimum. The whole
thing is time-boxed, so it always returns the best result found so far
rather than failing on a hard instance.

**No stale state between benchmarks.** The solver is stateless across
calls. No caching, no tuning on the public names. Everything you see in
the table reproduces from a fresh process with seed=0 and a 20-second
budget.

---

## What is proven, and what is still open

Four of six benchmarks are proven optimal:

- **ghz_star — 6.5.** The program wants one hub qubit connected to seven
  leaves. The chip's maximum physical degree is three, so no qubit can sit
  next to seven others. That forces at least two SWAPs, and the depth floor
  is 9. Two SWAPs plus half of 9 is exactly 6.5. Our solution matches.
- **chain_trotter — 4.5.** Fits on the Hamiltonian path. Depth 9, zero
  SWAPs. Optimal.
- **ladder_trotter — 6.5.** Three SWAPs, depth 7, matching a bound the
  solver computes during search.
- **vqe_layers — 3.0.** Also fits on the Hamiltonian path. Depth 6, zero
  SWAPs. Optimal.

Two benchmarks are best-found but not proven:

- **qaoa_random — 11.5.** The solver's internal floor is 9.0 (at least
  five SWAPs are required, and the critical path is depth 8). The remaining
  gap is at most 2.5 points. We do not know whether 11.0 is achievable.
- **dense_random — 36.0.** Forty interactions across fourteen logical
  qubits. The floor is 17.0 (at least eleven SWAPs, depth at least 12). Our
  result is 24 SWAPs at depth 23. A 240-trial placement sweep and a
  randomized-lineage sweep both plateau at exactly this point, which tells
  us that this particular search method is exhausted — but it does not
  tell us how close 36.0 is to the true minimum. The true minimum sits
  somewhere in [17.0, 36.0], and we have not closed that interval.

We report these gaps rather than filling them with optimism. If a reader
wants to know how much room is left on dense_random, the honest answer is:
at most 19 points, and we don't know how much less than that.

---

## Reproducing

From Computational Track/:

    python - <<'PY'
    import sys; sys.path.insert(0, ".")
    from solution import solve
    from starter_kit import BENCHMARKS, build_hardware_graph, score_summary

    hw = build_hardware_graph()
    total = 0.0
    for name, prog in BENCHMARKS.items():
        placement, routed = solve(prog, hw, time_budget=20, seed=0)
        result = score_summary(prog, hw, placement, routed)
        assert result["valid"], f"{name} produced an invalid routing"
        s = result["score"]
        total += s
        print(f"{name:16s}  score={s:6.1f}")
    print(f"{'TOTAL':16s}  score={total:6.1f}")
    PY

Default seed 0, default 20-second budget per benchmark. Runs on a single
CPU. On a heavily-loaded machine dense_random can stop at 37.0 instead of
36.0 (the search is anytime and stops when the clock runs out); the other
five benchmarks are deterministic and always give the same numbers.

Every result is validated against the official score_summary before it
is accepted. If a candidate routing fails the validity check, the solver
discards it and falls back to the last known-good routing. There is no path
through the code that submits an invalid routing.

---

## Repository layout

    Computational Track/
    ├── solution/
    │   ├── __init__.py       # exposes solve() at package level
    │   └── solve.py          # the full solver
    ├── starter_kit/          # provided by the organizers, unmodified
    ├── README.md             # starter-kit readme
    └── WRITEUP.md            # this document

The graded entry point is:

    from solution import solve
    placement, routed = solve(program, hardware_graph)

placement is a dict mapping logical qubit index → physical qubit index.
routed is a chronological list of ("1Q", p), ("2Q", p, q), and
("SWAP", p, q) tuples, all on physical qubits.

---

## Credits

The solver, the placement logic, the routing beam, and every result in this
document were developed for this submission. The starter kit — benchmark
programs, hardware graph, and the official scorer — is the organizers'
unmodified code.

The scores reported here are the only scores we claim. We did not score
against private benchmarks, and we did not tune against anything the
starter kit does not provide.

---

*Questions, corrections, or a case where our routing can be beaten? Open an
issue on the repository. We'll take a look.*

## Demo

A 4-minute walkthrough: https://youtu.be/_oUCP7Zyut4
