import random

import networkx as nx

from starter_kit import (
    BENCHMARKS,
    baseline_solve,
    build_hardware_graph,
    score_summary,
)


def _select_workable_component(hardware_graph, logical_qubits):
    components = sorted(nx.connected_components(hardware_graph), key=len, reverse=True)
    needed = len(logical_qubits)
    if not components:
        raise ValueError("hardware_graph has no nodes.")
    for comp in components:
        if len(comp) >= needed:
            return hardware_graph.subgraph(comp).copy()
    raise ValueError(
        f"Program needs {needed} qubits, but the largest connected region of "
        f"hardware_graph only has {len(components[0])} mutually reachable qubits. "
        f"This program cannot be routed on this hardware graph."
    )


def _order_by_weight(program):
    interaction_weight = {}
    for op in program:
        if op[0] == "2Q":
            _, i, j = op
            key = tuple(sorted((i, j)))
            interaction_weight[key] = interaction_weight.get(key, 0) + 1
    logical_qubits = sorted({q for op in program for q in op[1:]})
    total_weight = {q: 0 for q in logical_qubits}
    for (i, j), w in interaction_weight.items():
        total_weight[i] += w
        total_weight[j] += w
    order = sorted(logical_qubits, key=lambda q: -total_weight[q])
    return order, interaction_weight


def _order_by_bfs_clustering(program, interaction_weight):
    logical_qubits = sorted({q for op in program for q in op[1:]})
    adj = {q: [] for q in logical_qubits}
    for (i, j), w in interaction_weight.items():
        adj[i].append((j, w))
        adj[j].append((i, w))
    for q in adj:
        adj[q].sort(key=lambda x: -x[1])

    total_weight = {q: 0 for q in logical_qubits}
    for (i, j), w in interaction_weight.items():
        total_weight[i] += w
        total_weight[j] += w
    start = max(logical_qubits, key=lambda q: total_weight[q])

    visited = {start}
    order = [start]
    frontier = [start]
    while len(order) < len(logical_qubits):
        if not frontier:
            remaining = [q for q in logical_qubits if q not in visited]
            frontier = [max(remaining, key=lambda q: total_weight[q])]
        node = frontier.pop(0)
        for nbr, _ in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                order.append(nbr)
                frontier.append(nbr)
    return order


def _weighted_placement(order, interaction_weight, dist, physical_nodes, seed):
    placement = {order[0]: seed}
    used = {seed}
    for q in order[1:]:
        best_p, best_score = None, None
        for p in physical_nodes:
            if p in used:
                continue
            score = sum(
                interaction_weight.get(tuple(sorted((q, pq))), 0) * dist[p][placement[pq]]
                for pq in placement
            )
            if best_score is None or score < best_score:
                best_score, best_p = score, p
        placement[q] = best_p
        used.add(best_p)
    return placement


def _two_opt_refine_placement(placement, interaction_weight, dist, max_passes=15):
    logical_qubits = list(placement.keys())

    def total_cost(pl):
        return sum(w * dist[pl[i]][pl[j]] for (i, j), w in interaction_weight.items())

    current = dict(placement)
    current_cost = total_cost(current)

    for _ in range(max_passes):
        improved = False
        for ai in range(len(logical_qubits)):
            for bi in range(ai + 1, len(logical_qubits)):
                i, j = logical_qubits[ai], logical_qubits[bi]
                candidate = dict(current)
                candidate[i], candidate[j] = current[j], current[i]
                candidate_cost = total_cost(candidate)
                if candidate_cost < current_cost - 1e-9:
                    current = candidate
                    current_cost = candidate_cost
                    improved = True
        if not improved:
            break

    return current


def _route_sabre(program, hardware_graph, placement, dist, rng=None,
                  window=8, decay_step=0.05, decay_reset_every=15,
                  max_iters_factor=6, jitter=1e-6):
    current = dict(placement)
    phys_to_log = {p: l for l, p in current.items()}
    routed = []
    diameter_cap = max_iters_factor * (nx.diameter(hardware_graph) + 1)
    swap_decay = {p: 0.0 for p in hardware_graph.nodes}
    since_reset = 0

    for t, op in enumerate(program):
        if op[0] == "1Q":
            routed.append(("1Q", current[op[1]]))
            continue

        _, ll, lr = op
        lookahead = []
        w = 1.0
        cnt = 0
        for fut in program[t + 1:]:
            if fut[0] != "2Q":
                continue
            lookahead.append((fut[1], fut[2], w))
            w *= 0.5
            cnt += 1
            if cnt >= window:
                break

        iters = 0
        while not hardware_graph.has_edge(current[ll], current[lr]):
            pl, pr = current[ll], current[lr]
            candidate_edges = set()
            for node in (pl, pr):
                for nbr in hardware_graph.neighbors(node):
                    candidate_edges.add((node, nbr))

            best_edge, best_key = None, None
            for (a, b) in candidate_edges:
                a_log, b_log = phys_to_log.get(a), phys_to_log.get(b)
                new_current = dict(current)
                if a_log is not None:
                    new_current[a_log] = b
                if b_log is not None:
                    new_current[b_log] = a

                cost = dist[new_current[ll]][new_current[lr]]
                for (li, lj, wt) in lookahead:
                    cost += wt * dist[new_current[li]][new_current[lj]]
                cost += swap_decay[a] + swap_decay[b]

                if rng is not None:
                    cost += rng.uniform(0, jitter)

                if best_key is None or cost < best_key:
                    best_key, best_edge = cost, (a, b)

            a, b = best_edge
            a_log, b_log = phys_to_log.get(a), phys_to_log.get(b)
            routed.append(("SWAP", a, b))
            phys_to_log[a], phys_to_log[b] = b_log, a_log
            if a_log is not None:
                current[a_log] = b
            if b_log is not None:
                current[b_log] = a

            swap_decay[a] += decay_step
            swap_decay[b] += decay_step
            since_reset += 1
            if since_reset >= decay_reset_every:
                swap_decay = {p: 0.0 for p in hardware_graph.nodes}
                since_reset = 0

            iters += 1
            if iters > diameter_cap:
                path = nx.shortest_path(hardware_graph, current[ll], current[lr])
                a, b = path[0], path[1]
                a_log, b_log = phys_to_log.get(a), phys_to_log.get(b)
                routed.append(("SWAP", a, b))
                phys_to_log[a], phys_to_log[b] = b_log, a_log
                if a_log is not None:
                    current[a_log] = b
                if b_log is not None:
                    current[b_log] = a

        routed.append(("2Q", current[ll], current[lr]))

    return routed


def _route_sabre_with_final(program, hardware_graph, placement, dist, rng=None, window=8):
    routed = _route_sabre(program, hardware_graph, placement, dist, rng=rng, window=window)
    current = dict(placement)
    phys_to_log = {p: l for l, p in current.items()}
    for op in routed:
        if op[0] == "SWAP":
            a, b = op[1], op[2]
            a_log, b_log = phys_to_log.get(a), phys_to_log.get(b)
            phys_to_log[a], phys_to_log[b] = b_log, a_log
            if a_log is not None:
                current[a_log] = b
            if b_log is not None:
                current[b_log] = a
    return routed, current


def _cancel_redundant_swaps(routed_program):
    changed = True
    ops = list(routed_program)
    while changed:
        changed = False
        i = 0
        new_ops = []
        while i < len(ops):
            if (
                i + 1 < len(ops)
                and ops[i][0] == "SWAP"
                and ops[i + 1][0] == "SWAP"
                and {ops[i][1], ops[i][2]} == {ops[i + 1][1], ops[i + 1][2]}
            ):
                i += 2
                changed = True
            else:
                new_ops.append(ops[i])
                i += 1
        ops = new_ops
    return ops


def solve(program, hardware_graph, restarts_per_seed=3, base_seed=42, window=8):
    """
    Robustness: restricts work to a connected region of hardware_graph big
    enough for the program; raises a clear ValueError if unroutable.

    Placement: two orderings (busiest-first, BFS clustering) x every
    physical seed, each refined by 2-opt local search (QAP heuristic),
    then one proper SABRE bidirectional pass (forward, backward, forward).

    Routing: SABRE-style look-ahead with decay and randomized restarts,
    post-processed to cancel adjacent redundant SWAP pairs.
    """
    order_weighted, interaction_weight = _order_by_weight(program)
    order_bfs = _order_by_bfs_clustering(program, interaction_weight)
    work_graph = _select_workable_component(hardware_graph, order_weighted)

    dist = dict(nx.all_pairs_shortest_path_length(work_graph))
    physical_nodes = list(work_graph.nodes)
    reversed_program = list(reversed(program))

    best_score, best_result = None, None
    attempt = 0

    for order in (order_weighted, order_bfs):
        for seed in physical_nodes:
            greedy_placement = _weighted_placement(order, interaction_weight, dist, physical_nodes, seed)
            placement = _two_opt_refine_placement(greedy_placement, interaction_weight, dist)

            for _ in range(restarts_per_seed):
                rng = random.Random(base_seed + attempt)
                attempt += 1

                _, mid_placement = _route_sabre_with_final(
                    program, work_graph, placement, dist, rng=rng, window=window
                )
                _, mid_placement2 = _route_sabre_with_final(
                    reversed_program, work_graph, mid_placement, dist, rng=rng, window=window
                )
                routed, _ = _route_sabre_with_final(
                    program, work_graph, mid_placement2, dist, rng=rng, window=window
                )

                routed = _cancel_redundant_swaps(routed)

                result = score_summary(program, hardware_graph, mid_placement2, routed)
                if not result["valid"]:
                    continue
                if best_score is None or result["score"] < best_score:
                    best_score, best_result = result["score"], (mid_placement2, routed)

    return best_result


def _random_program(num_qubits, num_gates, seed):
    rng = random.Random(seed)
    prog = []
    for _ in range(num_gates):
        a, b = rng.sample(range(num_qubits), 2)
        prog.append(("2Q", a, b))
    return prog


def _generalization_suite():
    return {
        "fresh_small": _random_program(8, 12, seed=101),
        "fresh_medium": _random_program(12, 24, seed=202),
        "fresh_dense": _random_program(16, 50, seed=303),
        "fresh_sparse": _random_program(18, 15, seed=404),
        "fresh_large": _random_program(20, 60, seed=505),
    }


def _robustness_suite():
    print("\nROBUSTNESS CHECK (edge cases)")
    print("-" * 60)

    disconnected = nx.Graph()
    disconnected.add_edges_from([(0, 1), (1, 2), (2, 3)])
    disconnected.add_edges_from([(10, 11), (11, 12), (12, 13)])
    small_program = [("2Q", 0, 1), ("2Q", 1, 2), ("2Q", 2, 3)]
    try:
        result = solve(small_program, disconnected)
        pl, rt = result
        check = score_summary(small_program, disconnected, pl, rt)
        print(f"1. Disconnected graph, program fits one island: "
              f"{'PASS' if check['valid'] else 'FAIL'} (score {check['score']:.1f})")
    except Exception as exc:
        print(f"1. Disconnected graph, program fits one island: CRASHED -> {exc}")

    too_big_program = [("2Q", i, i + 1) for i in range(7)]
    try:
        solve(too_big_program, disconnected)
        print("2. Program too big for any single island: did NOT raise an error (unexpected)")
    except ValueError as exc:
        print(f"2. Program too big for any single island: PASS -> raised clear error: {exc}")
    except Exception as exc:
        print(f"2. Program too big for any single island: CRASHED with wrong error type -> {exc}")

    tiny_graph = nx.path_graph(3)
    oversized_program = [("2Q", i, i + 1) for i in range(9)]
    try:
        solve(oversized_program, tiny_graph)
        print("3. Program bigger than entire hardware graph: did NOT raise an error (unexpected)")
    except ValueError as exc:
        print(f"3. Program bigger than entire hardware graph: PASS -> raised clear error: {exc}")
    except Exception as exc:
        print(f"3. Program bigger than entire hardware graph: CRASHED with wrong error type -> {exc}")


if __name__ == "__main__":
    GRAPH = build_hardware_graph()

    def _run_suite(title, programs):
        print(f"\n{title}")
        print(f"{'benchmark':15s}  {'baseline':>9s}  {'yours':>9s}  {'delta':>8s}  {'%improve':>9s}")
        print("-" * 60)
        total_baseline = 0
        total_yours = 0
        for name, prog in programs.items():
            bl_pl, bl_rt = baseline_solve(prog, GRAPH)
            bl_res = score_summary(prog, GRAPH, bl_pl, bl_rt)

            result = solve(prog, GRAPH)
            if result is None:
                print(f"{name:15s}  {bl_res['score']:9.1f}  {'FAILED':>9s}")
                continue
            my_pl, my_rt = result
            my_res = score_summary(prog, GRAPH, my_pl, my_rt)

            total_baseline += bl_res["score"]
            total_yours += my_res["score"] if my_res["valid"] else 0

            tag = "OK" if my_res["valid"] else "INVALID"
            delta = my_res["score"] - bl_res["score"] if my_res["valid"] else float("nan")
            pct = (100 * delta / bl_res["score"]) if bl_res["score"] > 0 else 0
            print(f"{name:15s}  {bl_res['score']:9.1f}  {my_res['score']:9.1f}  {delta:8.1f}  {pct:8.1f}%  {tag}")

        print("-" * 60)
        pct_total = (100 * (total_yours - total_baseline) / total_baseline) if total_baseline > 0 else 0
        print(f"{'TOTAL':15s}  {total_baseline:9.1f}  {total_yours:9.1f}  {'':8s}  {pct_total:8.1f}%")
        return pct_total

    known_pct = _run_suite("KNOWN BENCHMARKS (used during development)", BENCHMARKS)
    fresh_pct = _run_suite("FRESH BENCHMARKS (never seen during tuning)", _generalization_suite())

    print(f"\n{'='*60}")
    print(f"Known benchmarks improvement:  {known_pct:6.1f}%")
    print(f"Fresh benchmarks improvement:  {fresh_pct:6.1f}%")
    gap = known_pct - fresh_pct
    print(f"Gap: {gap:5.1f} percentage points", end="  ")
    if abs(gap) < 10:
        print("-> looks fine, not meaningfully overfit")
    elif abs(gap) < 20:
        print("-> some gap, worth a closer look but not alarming")
    else:
        print("-> significant gap, likely overfit to known benchmarks")

    _robustness_suite()