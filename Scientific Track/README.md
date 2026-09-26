# Scientific Track — ANNNI Phase Diagram Under Noise

Full method, results, and analysis are in [WRITEUP.md](./WRITEUP.md).

## Run

    python analysis.py

## Requirements

PennyLane 0.44.1, NumPy, Matplotlib, NetworkX (see `pyproject.toml`).

## Output

Five figures are written to `outputs/`: the clean (p=0) phase diagram and
its raw order-parameter panels, the p=0.01 and p=0.05 phase diagrams, and
a boundary-shift comparison plot. See WRITEUP.md Section 12 for details
on each.
