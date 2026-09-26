# ANNNI Phase Diagram Under Depolarizing Noise

**Team IonQ · GitHub: Athleity · Q-SITE 2026 Scientific Track**

## 1. Overview

We mapped the phase diagram of the 1D ANNNI model in the (κ, h) plane using
exact diagonalization for the noiseless case and a variational circuit
(VQE) with depolarizing noise for the noisy cases. The clean diagram
reproduces the expected three-phase structure (ferromagnetic, antiphase,
paramagnetic) and tracks the known analytical boundaries reasonably well
at N=8.

The honest headline is on the noise side. When we separated our VQE
circuit's own approximation error from the effect of the added depolarizing
noise, we found that at both parameter cuts we tested, the approximation
error was as large as or larger than the noise-induced shift itself. At
one cut the noise effect was not distinguishable from zero at our
resolution. We report this as a limitation of the method and system size
we used, not as a null result about the physics — a more converged
circuit would very likely reveal a real, resolvable shift.

## 2. The Model

The Hamiltonian is

H = -Σ ZᵢZᵢ₊₁ + κ Σ ZᵢZᵢ₊₂ - h Σ Xᵢ

with periodic boundary conditions. The first term favors aligned
neighboring spins (ferromagnetic order). The second term, with coefficient
κ ≥ 0, favors anti-aligned next-nearest neighbors and competes with the
first term once κ becomes appreciable. The third term is a transverse
field that favors spins polarized along X, independent of their neighbors.

At zero temperature the system sits in the ground state of H for whatever
(κ, h) is chosen, and that ground state's qualitative character changes
abruptly at certain (κ, h) — these are the phase boundaries we are mapping.
For κ < 0.5 the competition is mainly between ferromagnetic order and the
field, with an Ising-type transition. For κ > 0.5 an antiphase region
appears at low h, with a narrow floating phase separating it from the
paramagnetic region at higher h. We did not attempt to resolve the
floating phase separately; at N=8 it is expected to be difficult to see
(see Section 9).

## 3. Method — Exact Diagonalization

For the clean (p=0) diagram we built the dense 2^N × 2^N Hamiltonian matrix
directly (N=8, so 256×256) and diagonalized it with `numpy.linalg.eigh`,
taking the lowest-energy eigenvector as the ground state. This is exact,
not an approximation, and fast enough to run on a 28×28 grid over
κ ∈ [0, 1], h ∈ [0, 2] in about a minute.

## 4. Method — VQE

For the noisy diagrams we used a hardware-efficient ansatz: three layers,
each layer applying an RY rotation to every qubit followed by a ring of
CNOTs (qubit i to qubit (i+1) mod N). When noise is enabled, a
`DepolarizingChannel(p)` is placed on the target qubit immediately after
every CNOT, matching the noise model specified for this challenge.

For each (κ, h) grid point we optimized the ansatz's parameters once,
noiselessly, against the true Hamiltonian using Adam for 100 steps. We
then reused that single set of optimized parameters to evaluate order
parameters at three noise levels: p=0 (no added noise), p=0.01, and
p=0.05. We did this instead of optimizing separately for each noise level
for two reasons. First, it reflects the actual physical scenario the
challenge describes — a circuit is compiled once and then executed on
noisy hardware, rather than re-optimized for each noise level, which
would not be possible on real hardware in the first place. Second, it is
computationally cheaper: one optimization plus three cheap measurement
circuits, rather than three separate full optimizations.

The p=0 evaluation of this VQE circuit is not the same thing as the exact
ground state. It is the noiseless output of an imperfectly-converged
variational circuit, and we treat it separately in the analysis below for
exactly that reason.

## 5. Phase Classification

At each (κ, h) point we computed three order-parameter-like quantities
from the state (either the exact ground state or the VQE circuit's
output, depending on context):

- ⟨ZᵢZᵢ₊₁⟩, averaged over sites — large and positive in the
  ferromagnetic phase.
- −⟨ZᵢZᵢ₊₂⟩, averaged over sites — large and positive in the antiphase,
  where next-nearest neighbors anti-align.
- ⟨X⟩, averaged over sites — large and positive in the paramagnetic
  phase.

We classify each point by whichever of these three scores is largest. We
validated this against exact diagonalization at three unambiguous deep
points before using it on the full grid: κ=0, h=0.1 (deep ferro) gave
scores (0.997, −0.997, 0.050) and classified ferro; κ=1, h=0.05 (deep
antiphase) gave (0.000, 0.999, 0.025) and classified antiphase; κ=0.3,
h=2.0 (deep para) gave (0.207, 0.004, 0.952) and classified paramagnetic.
All three matched the expected phase.

## 6. Results — Clean Phase Diagram

The p=0 exact-diagonalization diagram (28×28 grid, `phase_diagram_clean.png`)
shows the three phases in their expected regions: ferromagnetic at low κ
and low h, antiphase at κ above roughly 0.5 and low h, and paramagnetic
occupying the rest. The ferro-para boundary tracks the analytical Ising
transition line reasonably closely for κ < 0.5. The antiphase-para
boundary sits between the KT and BKT reference lines, consistent with our
classification not separately resolving the floating phase that is
expected to occupy that thin sliver. `observable_panels_clean.png` shows
the three raw order-parameter quantities as continuous heatmaps rather
than the discretized three-color classification, which makes the
gradual, rather than sharp, nature of the finite-size crossover visible.

## 7. Results — VQE at Two Noise Levels

The p=0.01 and p=0.05 diagrams (15×15 grid, `phase_diagram_p001.png` and
`phase_diagram_p005.png`) were produced from the same VQE pipeline
described in Section 4. Visually, both resemble the clean diagram's
overall three-phase layout, but the ferromagnetic and antiphase regions
are somewhat smaller at low κ and low h than in the exact-diagonalization
diagram. Based on the boundary analysis in Section 8, we attribute most
of this difference to VQE approximation error rather than to the added
depolarizing noise itself, and we did not observe a clear, further
narrowing between p=0.01 and p=0.05 at the two cuts we checked in detail.

## 8. Boundary-Shift Analysis

We scanned h at two fixed κ values, at four settings each: exact
diagonalization, the noiseless VQE baseline, VQE at p=0.01, and VQE at
p=0.05, and recorded the h at which each first classified as paramagnetic.

**κ = 0.3 (ferromagnetic region)**

| Setting | h boundary | vs. exact | vs. VQE baseline |
|---|---|---|---|
| Exact diagonalization | 0.550 | — | — |
| VQE baseline (p=0) | 0.800 | +0.250 (method error) | — |
| VQE, p=0.01 | 0.750 | +0.200 | −0.050 (noise-only shift) |
| VQE, p=0.05 | 0.750 | +0.200 | −0.050 (noise-only shift) |

**κ = 0.7 (antiphase / floating region)**

| Setting | h boundary | vs. exact | vs. VQE baseline |
|---|---|---|---|
| Exact diagonalization | 0.500 | — | — |
| VQE baseline (p=0) | 0.550 | +0.050 (method error) | — |
| VQE, p=0.01 | 0.550 | +0.050 | 0.000 |
| VQE, p=0.05 | 0.550 | +0.050 | 0.000 |

## 9. Discussion — The Honest Result

At κ=0.3, the gap between exact diagonalization and our noiseless VQE
baseline is +0.250 in h. The measured noise-only effect, isolated from
that baseline, is −0.050 at both p=0.01 and p=0.05 — five times smaller
than our own method's approximation error, and identical at two different
noise strengths, which suggests we are not resolving a genuine p-dependent
effect there so much as hitting the resolution floor of our h-scan
combined with the VQE's approximation error.

At κ=0.7 the situation is more limited still: the noise-only shift is
exactly 0.000 at both noise levels. We do not read this as "the antiphase
region is immune to noise at this κ." We read it as: our VQE circuit,
optimizer, and grid resolution at N=8 are not precise enough to separate
a real noise effect from zero at this particular cut, given that even the
noiseless baseline already differs from the exact result by 0.050.

The overall, plainly stated conclusion is that our approximation error
exceeded the effect size we were trying to measure. This is a real
finding about the limits of this specific setup, not a claim about the
physics of the ANNNI model itself, and not something we found a way to
resolve within the time available for this submission.

## 10. What Would Close the Gap

A few concrete changes would likely let a future pass distinguish real
noise effects from VQE approximation error at this system size:

- **A deeper or more expressive ansatz.** Three layers of RY-plus-CNOT-ring
  is a modest circuit; more layers, or an ansatz shaped more closely to
  the Hamiltonian's own coupling structure, would likely close much of
  the 0.25-in-h gap we measured at κ=0.3.
- **More optimization steps, or a better-tuned optimizer.** 100 Adam steps
  is not large for a 24-parameter landscape; we saw no sign we had
  reached a true optimum rather than a reasonable but imperfect one.
- **Training under noise rather than transferring noiseless parameters.**
  We optimized noiselessly and then measured under noise, which is a
  reasonable model of "compile once, run on noisy hardware," but it
  means the optimizer never had the chance to find noise-robust
  parameters. Training the same ansatz directly against the noisy cost
  function at each p would give a fairer best case for what noise does
  to an appropriately-adapted circuit.
- **Larger N.** Finite-size effects are visible even in the exact
  diagonalization result (the crossover in the continuous panels is a
  gradient, not a step). N ≥ 12 would sharpen the boundaries themselves,
  independent of the VQE question, and is listed as a bonus target in
  the challenge for the same reason.

## 11. Reproducing

From the `Scientific Track` directory, with the environment set up per
`pyproject.toml` (PennyLane 0.44.1, NumPy, Matplotlib, NetworkX):

    python analysis.py

The clean exact-diagonalization diagram takes roughly a minute. The
combined VQE pass over both noisy diagrams and the boundary-shift scan
together took approximately 20-25 minutes in our run. All output is
written to `outputs/`.

## 12. Figures in `outputs/`

- `phase_diagram_clean.png` — discrete three-phase classification, p=0,
  exact diagonalization, 28×28 grid, with analytical boundary overlays.
- `observable_panels_clean.png` — the three raw order-parameter values
  (nearest ZZ, negative next-nearest ZZ, X) as continuous heatmaps over
  the same clean grid.
- `phase_diagram_p001.png` — discrete classification from the VQE
  circuit at p=0.01, 15×15 grid.
- `phase_diagram_p005.png` — the same, at p=0.05.
- `noise_boundary_shift.png` — the exact/baseline/p=0.01/p=0.05 boundary
  values from Section 8 at both κ cuts, plotted together to show method
  error and noise-only shift side by side.

## 13. Layout

    Scientific Track/
      analysis.py
      WRITEUP.md
      README.md
      pyproject.toml
      starter_kit/
        annni.py
        exact_diag.py
        observables.py
        noise_utils.py
        plotting.py
        reference.py
      outputs/
        phase_diagram_clean.png
        observable_panels_clean.png
        phase_diagram_p001.png
        phase_diagram_p005.png
        noise_boundary_shift.png

## 14. Summary

We built and validated a phase-classification method on the ANNNI model,
confirmed it against exact diagonalization at deep, unambiguous points,
and used it to produce a clean phase diagram that matches the expected
analytical structure. We then extended the same method to a VQE circuit
under depolarizing noise at two levels, and attempted to isolate the
noise-induced shift in the phase boundary from the circuit's own
approximation error by comparing against a noiseless VQE baseline rather
than against the exact result directly. At both parameter cuts we
checked, our approximation error was comparable to or larger than the
noise effect we were trying to measure, which we report as the limiting
factor of this submission rather than as a conclusion about the physics.
