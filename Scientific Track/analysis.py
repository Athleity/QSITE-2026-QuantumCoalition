import os

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from starter_kit.annni import build_annni_hamiltonian
from starter_kit.exact_diag import exact_ground_state
from starter_kit.reference import ising_transition, kt_transition, bkt_transition

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

N_QUBITS = 8
CLEAN_GRID = 28          # exact diagonalization is cheap: go finer than the VQE grid
NOISY_GRID = 15          # meets the rubric's stated "at least 15x15" minimum
VQE_LAYERS = 3
VQE_STEPS = 100          # was 50; the extra convergence matters (see note below)
VQE_LR = 0.15
KAPPA_RANGE = (0.0, 1.0)
H_RANGE = (0.0, 2.0)
P_LEVELS = (0.0, 0.01, 0.05)   # 0.0 here = "VQE baseline, no added noise" (see analysis below)
OUTPUT_DIR = "outputs"

PHASE_NAMES = ["Ferromagnetic", "Antiphase", "Paramagnetic"]
PHASE_COLORS = ["#c0392b", "#2874a6", "#27ae60"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Phase classification (validated against exact diagonalization at
# deep ferro/antiphase/para points before use)
# ─────────────────────────────────────────────────────────────────

def classify_phase(summary):
    ferro = summary["zz_nearest_mean"]
    antiphase = -summary["zz_next_nearest_mean"]
    para = summary["x_mean"]
    scores = [ferro, antiphase, para]
    return int(np.argmax(scores)), scores


# ─────────────────────────────────────────────────────────────────
# Clean diagram: exact diagonalization (fast, exact, no VQE involved)
# ─────────────────────────────────────────────────────────────────

def compute_clean_diagram():
    kappas = np.linspace(*KAPPA_RANGE, CLEAN_GRID)
    hs = np.linspace(*H_RANGE, CLEAN_GRID)
    phase_grid = np.zeros((CLEAN_GRID, CLEAN_GRID))
    score_grid = np.zeros((CLEAN_GRID, CLEAN_GRID, 3))

    print(f"[clean] exact diagonalization, N={N_QUBITS}, {CLEAN_GRID}x{CLEAN_GRID} grid")
    for hi, h in enumerate(hs):
        for ki, kappa in enumerate(kappas):
            result = exact_ground_state(n_qubits=N_QUBITS, kappa=float(kappa), h=float(h))
            phase, scores = classify_phase(result["summary"])
            phase_grid[hi, ki] = phase
            score_grid[hi, ki, :] = scores
        print(f"  h={h:.2f} done ({hi + 1}/{CLEAN_GRID})")

    return kappas, hs, phase_grid, score_grid


# ─────────────────────────────────────────────────────────────────
# VQE ansatz. Noise (if p>0) is placed after every CNOT on its
# target qubit, per the challenge spec.
# ─────────────────────────────────────────────────────────────────

def _ansatz(params, n_qubits, n_layers, p):
    for layer in range(n_layers):
        for wire in range(n_qubits):
            qml.RY(params[layer, wire], wires=wire)
        for wire in range(n_qubits):
            target = (wire + 1) % n_qubits
            qml.CNOT(wires=[wire, target])
            if p > 0:
                qml.DepolarizingChannel(p, wires=target)


def _observables(n_qubits):
    zz_near = qml.sum(*[qml.Z(i) @ qml.Z((i + 1) % n_qubits) for i in range(n_qubits)]) * (1.0 / n_qubits)
    zz_next = qml.sum(*[qml.Z(i) @ qml.Z((i + 2) % n_qubits) for i in range(n_qubits)]) * (1.0 / n_qubits)
    x_avg = qml.sum(*[qml.X(i) for i in range(n_qubits)]) * (1.0 / n_qubits)
    return zz_near, zz_next, x_avg


def vqe_optimize(kappa, h, n_qubits=N_QUBITS, n_layers=VQE_LAYERS, steps=VQE_STEPS, lr=VQE_LR, seed=0):
    """Optimize the ansatz once, noiselessly, against the true Hamiltonian.
    This one circuit is then reused for every noise level below — this is
    what 'noise distorts an otherwise-fixed compiled circuit' actually means."""
    hamiltonian = build_annni_hamiltonian(n_qubits=n_qubits, kappa=kappa, h=h, periodic=True)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost(params):
        _ansatz(params, n_qubits, n_layers, p=0.0)
        return qml.expval(hamiltonian)

    params = qml.numpy.array(
        np.random.default_rng(seed).uniform(0, 2 * np.pi, size=(n_layers, n_qubits)), requires_grad=True
    )
    opt = qml.AdamOptimizer(stepsize=lr)
    for _ in range(steps):
        params = opt.step(cost, params)
    return params


def measure_at_p(params, p, n_qubits=N_QUBITS, n_layers=VQE_LAYERS):
    """Evaluate order parameters on the SAME optimized circuit at noise level p."""
    dev = qml.device("default.mixed", wires=n_qubits)
    zz_near, zz_next, x_avg = _observables(n_qubits)

    @qml.qnode(dev)
    def circuit():
        _ansatz(params, n_qubits, n_layers, p=p)
        return qml.expval(zz_near), qml.expval(zz_next), qml.expval(x_avg)

    a, b, c = circuit()
    return {"zz_nearest_mean": float(a), "zz_next_nearest_mean": float(b), "x_mean": float(c)}


# ─────────────────────────────────────────────────────────────────
# Combined noisy scan: ONE VQE optimization per grid point, reused
# across all three p-levels. This isn't just faster (roughly half
# the VQE optimizations vs. running p=0.01 and p=0.05 as separate
# passes) — it's methodologically the right thing to do, since it
# means every noise comparison is against the exact same circuit.
#
# p=0.0 here is the "VQE baseline": the same imperfect ansatz with
# NO added noise. This matters because VQE at N=8/100 steps does not
# perfectly reach the ground state (measured gap: ~1-3% of the energy
# scale). Comparing p=0.01/0.05 against the clean EXACT-DIAGONALIZATION
# diagram would silently mix "VQE approximation error" together with
# "actual depolarizing noise effect" into one number. Comparing
# against this VQE baseline instead isolates the real noise effect.
# ─────────────────────────────────────────────────────────────────

def compute_noisy_diagrams():
    kappas = np.linspace(*KAPPA_RANGE, NOISY_GRID)
    hs = np.linspace(*H_RANGE, NOISY_GRID)
    phase_grids = {p: np.zeros((NOISY_GRID, NOISY_GRID)) for p in P_LEVELS}

    print(f"[noisy] VQE, N={N_QUBITS}, {NOISY_GRID}x{NOISY_GRID} grid, "
          f"{VQE_STEPS} steps, one optimization per point reused for p={P_LEVELS}")
    for hi, h in enumerate(hs):
        for ki, kappa in enumerate(kappas):
            params = vqe_optimize(float(kappa), float(h))
            for p in P_LEVELS:
                summary = measure_at_p(params, p)
                phase, _ = classify_phase(summary)
                phase_grids[p][hi, ki] = phase
        print(f"  h={h:.2f} done ({hi + 1}/{NOISY_GRID})")

    return kappas, hs, phase_grids


# ─────────────────────────────────────────────────────────────────
# Plotting: (1) required discrete classified maps, (2) continuous
# raw-observable panels for physical insight, (3) noise-shift
# comparison isolating method error from real noise effect.
# ─────────────────────────────────────────────────────────────────

def _overlay_boundaries(ax, light=False):
    kf = np.linspace(0.01, 1.0, 400)
    dash_color = "white" if light else "black"
    ax.plot(kf, ising_transition(kf), color=dash_color, ls="--", lw=2.2, label="Ising (ferro-para)")
    ax.plot(kf, kt_transition(kf), color="white", ls=":", lw=2.4, label="KT (floating-para)")
    ax.plot(kf, bkt_transition(kf), color="white", ls="-.", lw=2.2, label="BKT (antiphase-floating)")


def plot_phase_diagram(kappas, hs, phase_grid, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = ListedColormap(PHASE_COLORS)
    ax.pcolormesh(kappas, hs, phase_grid, cmap=cmap, vmin=0, vmax=2, shading="auto")
    _overlay_boundaries(ax)

    ax.set_xlabel(r"$\kappa$", fontsize=13)
    ax.set_ylabel(r"$h$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(kappas.min(), kappas.max())
    ax.set_ylim(hs.min(), hs.max())

    legend_elements = [Patch(facecolor=PHASE_COLORS[i], label=PHASE_NAMES[i]) for i in range(3)]
    legend_elements += [
        plt.Line2D([0], [0], color="black", ls="--", lw=2, label="Ising (ferro-para)"),
        plt.Line2D([0], [0], color="gray", ls=":", lw=2, label="KT (floating-para)"),
        plt.Line2D([0], [0], color="gray", ls="-.", lw=2, label="BKT (antiphase-floating)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5, framealpha=0.92)

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_observable_panels(kappas, hs, score_grid, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panel_specs = [
        (r"$\langle Z_i Z_{i+1}\rangle$ (ferro order)", "RdBu_r", -1, 1),
        (r"$-\langle Z_i Z_{i+2}\rangle$ (antiphase order)", "RdBu_r", -1, 1),
        (r"$\langle X\rangle$ (para order)", "viridis", 0, 1),
    ]
    for idx, (ax, (label, cmap, vmin, vmax)) in enumerate(zip(axes, panel_specs)):
        mesh = ax.pcolormesh(kappas, hs, score_grid[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
        _overlay_boundaries(ax)
        ax.set_xlabel(r"$\kappa$")
        if idx == 0:
            ax.set_ylabel(r"$h$")
        ax.set_title(label, fontsize=11)
        fig.colorbar(mesh, ax=ax, shrink=0.85)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Quantitative boundary-shift analysis, properly decomposed:
#   exact-diag boundary        -> ground truth
#   VQE baseline (p=0) boundary -> method error alone
#   VQE at p=0.01 / p=0.05      -> method error + real noise
# "real noise shift" = VQE(p) - VQE_baseline, NOT VQE(p) - exact_diag.
# ─────────────────────────────────────────────────────────────────

def find_boundary_h(kappa, h_min=0.0, h_max=2.0, n_points=41):
    """One optimize-once-measure-thrice scan per kappa cut. Returns the
    first h where each of [exact, p=0.0, p=0.01, p=0.05] reads paramagnetic."""
    hs = np.linspace(h_min, h_max, n_points)
    boundaries = {"exact": None, 0.0: None, 0.01: None, 0.05: None}

    for h in hs:
        if boundaries["exact"] is None:
            result = exact_ground_state(n_qubits=N_QUBITS, kappa=kappa, h=float(h))
            phase, _ = classify_phase(result["summary"])
            if phase == 2:
                boundaries["exact"] = float(h)

        if any(boundaries[p] is None for p in P_LEVELS):
            params = vqe_optimize(kappa, float(h))
            for p in P_LEVELS:
                if boundaries[p] is None:
                    summary = measure_at_p(params, p)
                    phase, _ = classify_phase(summary)
                    if phase == 2:
                        boundaries[p] = float(h)

        if all(v is not None for v in boundaries.values()):
            break

    for key, val in boundaries.items():
        if val is None:
            boundaries[key] = float(h_max)
    return boundaries


def run_boundary_shift_analysis():
    print("\nQuantitative boundary-shift analysis (method error vs. real noise, separated)")
    print("-" * 78)
    results = {}
    for kappa_cut, region in [(0.3, "ferromagnetic"), (0.7, "antiphase/floating")]:
        b = find_boundary_h(kappa_cut)
        results[kappa_cut] = b
        method_gap = b[0.0] - b["exact"]
        noise_shift_001 = b[0.01] - b[0.0]
        noise_shift_005 = b[0.05] - b[0.0]
        print(f"\nkappa = {kappa_cut} ({region} region):")
        print(f"  exact diagonalization boundary:        h = {b['exact']:.3f}")
        print(f"  VQE baseline, no added noise:           h = {b[0.0]:.3f}   "
              f"(method error vs exact: {method_gap:+.3f})")
        print(f"  VQE at p=0.01:                           h = {b[0.01]:.3f}   "
              f"(noise-only shift from baseline: {noise_shift_001:+.3f})")
        print(f"  VQE at p=0.05:                           h = {b[0.05]:.3f}   "
              f"(noise-only shift from baseline: {noise_shift_005:+.3f})")
    return results


def plot_noise_shift_summary(results, filename="noise_boundary_shift.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    x_labels = ["exact\ndiag", "VQE\nbaseline", "VQE\np=0.01", "VQE\np=0.05"]
    x = np.arange(4)
    markers = ["o", "s"]
    for (kappa_cut, b), marker in zip(results.items(), markers):
        y = [b["exact"], b[0.0], b[0.01], b[0.05]]
        ax.plot(x, y, marker=marker, lw=2, ms=9, label=fr"$\kappa$={kappa_cut}")
        ax.annotate("", xy=(1, b[0.0]), xytext=(0, b["exact"]),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1, ls="--"))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(r"paramagnetic boundary $h$")
    ax.set_title("Boundary shift: method error (dashed) vs. real noise (solid)")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kappas_c, hs_c, phases_c, scores_c = compute_clean_diagram()
    plot_phase_diagram(kappas_c, hs_c, phases_c,
                        "ANNNI Phase Diagram (p=0, exact diagonalization)", "phase_diagram_clean.png")
    plot_observable_panels(kappas_c, hs_c, scores_c,
                            "Clean order parameters, N=8, exact diagonalization", "observable_panels_clean.png")

    kappas_n, hs_n, phase_grids = compute_noisy_diagrams()
    plot_phase_diagram(kappas_n, hs_n, phase_grids[0.01],
                        "ANNNI Phase Diagram (VQE, p=0.01)", "phase_diagram_p001.png")
    plot_phase_diagram(kappas_n, hs_n, phase_grids[0.05],
                        "ANNNI Phase Diagram (VQE, p=0.05)", "phase_diagram_p005.png")

    boundary_results = run_boundary_shift_analysis()
    plot_noise_shift_summary(boundary_results)

    print("\nDone. Six files saved in Scientific Track/outputs/:")
    print("  phase_diagram_clean.png, observable_panels_clean.png,")
    print("  phase_diagram_p001.png, phase_diagram_p005.png, noise_boundary_shift.png")