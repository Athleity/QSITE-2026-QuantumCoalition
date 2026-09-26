#!/usr/bin/env python3
"""Layer-by-layer visualization of routed circuits.

For each requested benchmark:
  1. Run solution.solve() to get a routed program.
  2. Group ops into ASAP layers (disjoint qubits per layer).
  3. Render the first N layers side by side.

Op format from the solver:
    ("1Q",   p)         single-qubit gate
    ("2Q",   p, q)      two-qubit program gate
    ("SWAP", p, q)      inserted SWAP

Usage:
    python plot_layers.py
    python plot_layers.py ghz_star --layers 6
    python plot_layers.py ghz_star dense_random --layers 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solution import solve
from starter_kit import BENCHMARKS, build_hardware_graph


GRID_COORDS = {
    0: (0, 4), 1: (1, 4), 2: (2, 4), 3: (3, 4),
    4: (0, 3), 5: (1, 3), 6: (2, 3), 7: (3, 3),
    8: (0, 2), 9: (1, 2), 10: (2, 2), 11: (3, 2),
    12: (0, 1), 13: (1, 1), 14: (2, 1), 15: (3, 1),
    16: (0, 0), 17: (1, 0), 18: (2, 0), 19: (3, 0),
}

EDGE_COLOR = "#c8d3e0"
SWAP_COLOR = "#e74c3c"
GATE_COLOR = "#2c7fb8"
NODE_FACE = "#f2f6fb"
NODE_EDGE = "#8892a4"
NODE_TEXT = "#1f2933"


def op_name(op):
    return op[0]


def op_qubits(op):
    return tuple(op[1:])


def schedule_layers(ops):
    """Ready-time scheduling: the same depth the official scorer computes.

    Each two-qubit op starts at 1 + max(ready time of its two qubits).
    After the op runs, both qubits are marked busy until that time.
    Single-qubit ops do not contribute to depth and are skipped.

    Returns (layers, depth) where layers is a list of op-lists in time order.
    """
    ready = {}
    by_time = {}
    for op in ops:
        name = op[0]
        if name == "1Q":
            continue
        qs = op[1:]
        t = 1 + max(ready.get(q, 0) for q in qs)
        for q in qs:
            ready[q] = t
        by_time.setdefault(t, []).append(op)
    depth = max(ready.values()) if ready else 0
    layers = [by_time[t] for t in sorted(by_time)]
    return layers, depth


def draw_hardware(ax, hw):
    for a, b in hw.edges:
        x1, y1 = GRID_COORDS[a]
        x2, y2 = GRID_COORDS[b]
        ax.plot([x1, x2], [y1, y2], color=EDGE_COLOR, lw=2.0,
                solid_capstyle="round", zorder=1)


def draw_layer(ax, layer):
    for op in layer:
        name = op_name(op)
        qs = op_qubits(op)
        if name == "SWAP" and len(qs) == 2:
            a, b = qs
            x1, y1 = GRID_COORDS[a]
            x2, y2 = GRID_COORDS[b]
            ax.plot([x1, x2], [y1, y2], color=SWAP_COLOR, lw=6.5,
                    solid_capstyle="round", zorder=3)
        elif name == "2Q" and len(qs) == 2:
            a, b = qs
            x1, y1 = GRID_COORDS[a]
            x2, y2 = GRID_COORDS[b]
            ax.plot([x1, x2], [y1, y2], color=GATE_COLOR, lw=5.0,
                    solid_capstyle="round", zorder=2, alpha=0.85)


def draw_nodes(ax):
    for q, (x, y) in GRID_COORDS.items():
        ax.add_patch(Circle((x, y), 0.32, facecolor=NODE_FACE,
                            edgecolor=NODE_EDGE, linewidth=1.8, zorder=4))
        ax.text(x, y, str(q), ha="center", va="center", fontsize=10,
                color=NODE_TEXT, zorder=5)


def render(name, hw, routed, n_show, out_path):
    layers, depth = schedule_layers(routed)
    total_layers = len(layers)
    swaps = sum(1 for op in routed if op_name(op) == "SWAP")
    score = swaps + 0.5 * depth

    shown = min(n_show, total_layers)
    fig, axes = plt.subplots(1, shown, figsize=(4.3 * shown, 4.8))
    if shown == 1:
        axes = [axes]

    for k in range(shown):
        ax = axes[k]
        draw_hardware(ax, hw)
        draw_layer(ax, layers[k])
        draw_nodes(ax)
        ax.set_xlim(-0.8, 3.8)
        ax.set_ylim(-0.8, 4.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Layer {k + 1}", fontsize=13, pad=10)

    fig.suptitle(f"{name}  ·  first {shown} of {total_layers} layers  ·  "
                 f"score {score:.1f}  ({swaps} SWAPs)",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name:<16}  layers={total_layers:<3d}  swaps={swaps:<3d}  "
          f"score={score:5.1f}  ->  {out_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("benchmarks", nargs="*", default=None)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="demo_assets")
    args = p.parse_args()

    names = args.benchmarks if args.benchmarks else list(BENCHMARKS.keys())
    for n in names:
        if n not in BENCHMARKS:
            print(f"unknown benchmark: {n}")
            print("available:", ", ".join(BENCHMARKS))
            sys.exit(1)

    outdir = HERE / args.outdir
    outdir.mkdir(exist_ok=True)

    hw = build_hardware_graph()
    print()
    print(f"Rendering {len(names)} benchmark(s), first {args.layers} layers.")
    print()

    for name in names:
        prog = BENCHMARKS[name]
        _, routed = solve(prog, hw, time_budget=args.budget, seed=args.seed)
        render(name, hw, routed, args.layers, outdir / f"layers_{name}.png")

    print()
    print(f"Output: {outdir}")
    print()


if __name__ == "__main__":
    main()
