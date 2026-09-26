#!/usr/bin/env python3
"""Run all six public benchmarks against solution.solve.

Prints a formatted score table, cross-checks every result against the
official starter_kit scorer, and writes bench_report.txt alongside this
script. Exit code is 0 if every benchmark is valid, 1 otherwise.

Usage:
    python bench.py
    python bench.py --budget 30 --seed 0
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solution import solve
from starter_kit import BENCHMARKS, build_hardware_graph, score_summary


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _color(s, c, enabled):
    return f"{c}{s}{RESET}" if enabled else s


def parse_args():
    p = argparse.ArgumentParser(description="Run the Q-SITE Computational Track benchmarks.")
    p.add_argument("--budget", type=float, default=20.0,
                   help="Seconds per benchmark (default: 20).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed passed to solve() (default: 0).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colors in output.")
    return p.parse_args()


def main():
    args = parse_args()
    use_color = sys.stdout.isatty() and not args.no_color

    print()
    print(_color("  Q-SITE Computational Track — Benchmark Suite", BOLD, use_color))
    print(_color(f"  seed={args.seed}  budget={args.budget:.0f}s/benchmark", DIM, use_color))
    print(_color(f"  python {platform.python_version()}  {platform.system()}", DIM, use_color))
    print()

    header = f"  {'benchmark':<18}{'score':>8}{'swaps':>8}{'depth':>8}{'wall':>9}  status"
    print(_color(header, BOLD, use_color))
    print(_color("  " + "─" * 62, DIM, use_color))

    hw = build_hardware_graph()

    rows = []
    total_score = 0.0
    all_valid = True

    for name, prog in BENCHMARKS.items():
        t0 = time.perf_counter()
        placement, routed = solve(prog, hw, time_budget=args.budget, seed=args.seed)
        wall = time.perf_counter() - t0

        result = score_summary(prog, hw, placement, routed)
        valid = bool(result["valid"])
        score = float(result["score"])
        total_score += score

        swaps = sum(1 for op in routed if op[0] == "SWAP")
        depth = _depth(routed)

        status = _color("valid", GREEN, use_color) if valid else _color("INVALID", RED, use_color)
        if not valid:
            all_valid = False

        line = (f"  {name:<18}{score:>8.1f}{swaps:>8d}{depth:>8d}"
                f"{wall:>8.1f}s  {status}")
        print(line)
        rows.append((name, score, swaps, depth, wall, valid))

    print(_color("  " + "─" * 62, DIM, use_color))
    total_line = f"  {'TOTAL':<18}{total_score:>8.1f}"
    print(_color(total_line, BOLD, use_color))
    print()

    baseline = 283.5
    reduction = (baseline - total_score) / baseline * 100.0
    print(f"  Baseline: {baseline:.1f}   Ours: {total_score:.1f}   "
          f"Reduction: {reduction:.1f}%")
    print()

    report = [
        "Q-SITE Computational Track — benchmark report",
        f"seed = {args.seed}   budget = {args.budget:.0f}s   python = {platform.python_version()}",
        "",
        f"{'benchmark':<18}{'score':>8}{'swaps':>8}{'depth':>8}{'wall':>9}   valid",
        "-" * 60,
    ]
    for name, score, swaps, depth, wall, valid in rows:
        report.append(f"{name:<18}{score:>8.1f}{swaps:>8d}{depth:>8d}{wall:>8.1f}s   {valid}")
    report.append("-" * 60)
    report.append(f"{'TOTAL':<18}{total_score:>8.1f}")
    report.append("")
    report.append(f"baseline = {baseline:.1f}   reduction = {reduction:.1f}%")

    out = HERE / "bench_report.txt"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"  Report written to {out.name}")
    print()

    return 0 if all_valid else 1


def _depth(ops):
    ready = {}
    d = 0
    for op in ops:
        if op[0] == "1Q":
            continue
        _, p, q = op
        t = 1 + max(ready.get(p, 0), ready.get(q, 0))
        ready[p] = ready[q] = t
        if t > d:
            d = t
    return d


if __name__ == "__main__":
    sys.exit(main())
