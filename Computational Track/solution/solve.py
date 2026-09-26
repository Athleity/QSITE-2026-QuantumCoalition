"""qsite_router.py -- standalone qubit placement + SWAP routing (only needs networkx + the standard library).

Objective (same as the track):  score = #SWAP + 0.5 * depth, where depth is the ASAP layering of all 2Q/SWAP ops.

Design (independent of any other solver):
  * State is a plain logical->physical map (`pos`), its inverse (`occ`), per-physical ready times (`rdy`).
    No lazy placement: every search starts from a complete initial layout.
  * Routing one gate = "layered swap search": starting from the current node, each layer applies one SWAP that touches
    one of the two gate qubits and must not increase their distance (a bounded number of sideways moves is allowed);
    states are de-duplicated per layer and the search stops as soon as the pair is adjacent.
  * Beam search over the gate list; ranking = exact cost so far + decayed look-ahead of upcoming distances.
  * Iterated forward/backward passes ("relaxation chains"): run the beam forward, take its final map as the start of
    a beam over the *reversed* circuit, take that final map as a new start, and repeat.  Multi-fidelity schedule:
    many cheap narrow chains  ->  wider chains on the best layouts  ->  noisy chains around the best layouts.
  * Layout generators: grow-at-first-use, simulated annealing on weighted distances, longest embeddable prefix.
  * Zero-SWAP check by subgraph embedding (score is then provably 0.5 * critical path).
  * Every candidate is re-validated by an independent checker before it can become the answer.
"""
from __future__ import annotations

import heapq
import math
import random
import time
from collections import deque

INF = 10**6
DEFAULT_BUDGET = 20.0


# ======================================================================================================
# Hardware
# ======================================================================================================
class Chip:
    def __init__(self, graph):
        try:
            labels = sorted(graph.nodes)
        except TypeError:
            labels = list(graph.nodes)
        ix = {p: i for i, p in enumerate(labels)}
        n = len(labels)
        nb = [set() for _ in range(n)]
        for u, v in graph.edges:
            if u != v:
                nb[ix[u]].add(ix[v])
                nb[ix[v]].add(ix[u])
        self.labels, self.n = labels, n
        self.nbrset = nb
        self.nbr = [sorted(s) for s in nb]
        self.deg = [len(s) for s in nb]
        self.n_edges = sum(self.deg) // 2
        self.dist = [self._bfs(s) for s in range(n)]

    def _bfs(self, s):
        d = [INF] * self.n
        d[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in self.nbr[x]:
                if d[y] == INF:
                    d[y] = d[x] + 1
                    q.append(y)
        return d


# ======================================================================================================
# Search node + primitive moves
# ======================================================================================================
class Node:
    __slots__ = ("pos", "occ", "rdy", "nsw", "dep", "trail")


def _fork(nd):
    c = Node()
    c.pos = nd.pos[:]
    c.occ = nd.occ[:]
    c.rdy = nd.rdy[:]
    c.nsw = nd.nsw
    c.dep = nd.dep
    c.trail = nd.trail
    return c


def _swap(nd, x, y):
    ox, oy = nd.occ[x], nd.occ[y]
    nd.occ[x], nd.occ[y] = oy, ox
    if ox >= 0:
        nd.pos[ox] = y
    if oy >= 0:
        nd.pos[oy] = x
    r = nd.rdy
    t = (r[x] if r[x] > r[y] else r[y]) + 1
    r[x] = r[y] = t
    if t > nd.dep:
        nd.dep = t
    nd.nsw += 1


def _gate(nd, x, y):
    r = nd.rdy
    t = (r[x] if r[x] > r[y] else r[y]) + 1
    r[x] = r[y] = t
    if t > nd.dep:
        nd.dep = t


def _root(chip, layout):
    nd = Node()
    nd.pos = list(layout)
    nd.occ = [-1] * chip.n
    for q, p in enumerate(layout):
        nd.occ[p] = q
    nd.rdy = [0] * chip.n
    nd.nsw = nd.dep = 0
    nd.trail = None
    return nd


def _moves_of(nd):
    out = []
    t = nd.trail
    while t is not None:
        t, step = t
        out.append(step)
    out.reverse()
    return out


def _advance(chip, nd, a, b, slack, cap):
    """All ways to make gate (a, b) executable from `nd`, each returned as a child node with the gate executed."""
    D = chip.dist
    nbr = chip.nbr
    pa, pb = nd.pos[a], nd.pos[b]
    d0 = D[pa][pb]
    if d0 >= INF:
        return []
    if d0 == 1:
        c = _fork(nd)
        _gate(c, pa, pb)
        c.trail = (nd.trail, ())
        return [c]
    budget = d0 - 1 + slack
    done = []
    layer = [(nd, (), 0)]
    for step in range(budget):
        left = budget - step - 1
        nxt = {}
        for cur, mv, side in layer:
            xa, xb = cur.pos[a], cur.pos[b]
            dc = D[xa][xb]
            last = mv[-1] if mv else None
            for src, other in ((xa, xb), (xb, xa)):
                for y in nbr[src]:
                    if y == other:
                        continue
                    if last is not None and (src in last and y in last):
                        continue
                    dn = D[y][other]
                    if dn < dc:
                        ns = side
                    elif dn == dc and side < slack:
                        ns = side + 1
                    else:
                        continue
                    if dn - 1 > left:
                        continue
                    ch = _fork(cur)
                    _swap(ch, src, y)
                    m2 = mv + ((src, y),)
                    if dn == 1:
                        _gate(ch, ch.pos[a], ch.pos[b])
                        ch.trail = (nd.trail, m2)
                        done.append(ch)
                    else:
                        key = tuple(ch.pos)
                        old = nxt.get(key)
                        if old is None or ch.nsw * 2 + ch.dep < old[0].nsw * 2 + old[0].dep:
                            nxt[key] = (ch, m2, ns)
        layer = [(c, m, s) for c, m, s in nxt.values()]
        if not layer:
            break
    if len(done) > cap:
        done.sort(key=lambda c: (c.nsw + 0.5 * c.dep, c.rdy[c.pos[a]] + c.rdy[c.pos[b]]))
        done = done[:cap]
    return done


def beam_pass(chip, seq, layout, width, slack, deadline, rng, look=20, decay=0.6, alpha=1.0, noise=0.0, cap=24):
    """Beam search over gate list `seq` starting from `layout`.  Returns the best final Node (or None)."""
    D = chip.dist
    G = len(seq)
    ga = [g[0] for g in seq]
    gb = [g[1] for g in seq]
    wts = [decay**i for i in range(look + 1)]
    beam = [_root(chip, layout)]
    for k in range(G):
        a, b = ga[k], gb[k]
        merged = {}
        for nd in beam:
            for ch in _advance(chip, nd, a, b, slack, cap):
                key = tuple(ch.pos)
                cost = ch.nsw + 0.5 * ch.dep
                old = merged.get(key)
                if old is None or cost < old[0] or (cost == old[0] and sum(ch.rdy) < old[1]):
                    merged[key] = (cost, sum(ch.rdy), ch)
        if not merged:
            return None
        hi = min(G, k + 1 + look)
        scored = []
        for cost, _, ch in merged.values():
            pos = ch.pos
            fut = 0.0
            for i in range(k + 1, hi):
                d = D[pos[ga[i]]][pos[gb[i]]] - 1
                if d > 0:
                    fut += wts[i - k - 1] * d
            v = cost + alpha * fut
            if noise:
                v += noise * rng.random()
            scored.append((v, id(ch), ch))
        if len(scored) > width:
            scored = heapq.nsmallest(width, scored)
        beam = [s[2] for s in scored]
        if time.perf_counter() > deadline:
            return None
    return min(beam, key=lambda c: (c.nsw + 0.5 * c.dep, c.nsw))


# ======================================================================================================
# Independent scorer / checker
# ======================================================================================================
def tally(routed):
    last = {}
    depth = swaps = 0
    for op in routed:
        if op[0] == "1Q":
            continue
        if op[0] == "SWAP":
            swaps += 1
        t = 1 + max(last.get(op[1], 0), last.get(op[2], 0))
        last[op[1]] = last[op[2]] = t
        depth = max(depth, t)
    return swaps + 0.5 * depth


def check(program, graph, placement, routed):
    want = {q for op in program for q in op[1:]}
    if set(placement) != want:
        return False
    used = list(placement.values())
    if len(set(used)) != len(used) or any(p not in graph for p in used):
        return False
    holder = {p: q for q, p in placement.items()}
    seen = []
    for op in routed:
        if op[0] == "SWAP":
            _, x, y = op
            if not graph.has_edge(x, y):
                return False
            holder[x], holder[y] = holder.get(y), holder.get(x)
        elif op[0] == "2Q":
            _, x, y = op
            if not graph.has_edge(x, y) or holder.get(x) is None or holder.get(y) is None:
                return False
            seen.append(("2Q", holder[x], holder[y]))
        elif op[0] == "1Q":
            if holder.get(op[1]) is None:
                return False
            seen.append(("1Q", holder[op[1]]))
        else:
            return False
    return seen == [tuple(op) for op in program]


def naive_route(program, graph):
    """Always-valid fallback: identity layout + shortest-path swaps."""
    import networkx as nx

    logicals = sorted({q for op in program for q in op[1:]})
    try:
        phys = sorted(graph.nodes)
    except TypeError:
        phys = list(graph.nodes)
    if len(logicals) > len(phys):
        # No valid placement exists at all; nothing to route to. Return a partial, honestly-invalid
        # answer rather than raising, so a caller iterating over many instances doesn't crash.
        return {q: phys[i] for i, q in enumerate(logicals[: len(phys)])}, []
    place = {q: phys[i] for i, q in enumerate(logicals)}
    where = dict(place)
    holder = {p: q for q, p in where.items()}
    out = []
    for op in program:
        if op[0] == "1Q":
            out.append(("1Q", where[op[1]]))
            continue
        _, a, b = op
        if not graph.has_edge(where[a], where[b]):
            path = nx.shortest_path(graph, where[a], where[b])
            for x, y in zip(path[:-2], path[1:-1]):
                out.append(("SWAP", x, y))
                qx, qy = holder.get(x), holder.get(y)
                holder[x], holder[y] = qy, qx
                if qx is not None:
                    where[qx] = y
                if qy is not None:
                    where[qy] = x
        out.append(("2Q", where[a], where[b]))
    return place, out


# ======================================================================================================
# Layout generators
# ======================================================================================================
def complete_layout(chip, partial, L, rng):
    """Fill every -1 slot with a random free physical qubit."""
    lay = list(partial)
    have = {x for x in lay if x >= 0}
    free = [p for p in range(chip.n) if p not in have]
    rng.shuffle(free)
    for q in range(L):
        if lay[q] < 0:
            lay[q] = free.pop()
    return lay


def grow_layout(chip, seq, L, rng, partial=None):
    """Place each qubit next to its partner the first time it appears (random tie-breaking)."""
    lay = list(partial) if partial else [-1] * L
    taken = {p for p in lay if p >= 0}
    D = chip.dist
    for a, b in seq:
        for q, o in ((a, b), (b, a)):
            if lay[q] >= 0:
                continue
            free = [p for p in range(chip.n) if p not in taken]
            if not free:
                continue
            if lay[o] >= 0:
                dm = min(D[lay[o]][p] for p in free)
                cand = [p for p in free if D[lay[o]][p] <= dm + (1 if rng.random() < 0.3 else 0)]
            else:
                cand = [p for p in free if any(y not in taken for y in chip.nbr[p])] or free
            p = rng.choice(cand)
            lay[q] = p
            taken.add(p)
    return complete_layout(chip, lay, L, rng)


def anneal_layout(chip, seq, L, rng, decay=1.0, iters=2500):
    w = {}
    f = 1.0
    for a, b in seq:
        key = (a, b) if a < b else (b, a)
        w[key] = w.get(key, 0.0) + f
        f *= decay
    act = sorted({q for g in seq for q in g})
    if not act or len(act) > chip.n:
        return None
    adj = {q: [] for q in act}
    for (a, b), x in w.items():
        adj[a].append((b, x))
        adj[b].append((a, x))
    slots = list(range(chip.n))
    rng.shuffle(slots)
    where = {q: slots[i] for i, q in enumerate(act)}
    who = [-1] * chip.n
    for q, p in where.items():
        who[p] = q
    D = chip.dist

    def pull(q, p):
        return sum(x * D[p][where[m]] for m, x in adj[q])

    cost = sum(x * D[where[a]][where[b]] for (a, b), x in w.items())
    best, best_where = cost, dict(where)
    hi = max(w.values()) * 2.0
    lo = hi * 0.005
    for it in range(iters):
        temp = hi * (lo / hi) ** (it / iters)
        q = act[rng.randrange(len(act))]
        p0, p1 = where[q], rng.randrange(chip.n)
        if p0 == p1:
            continue
        m = who[p1]
        before = pull(q, p0) + (pull(m, p1) if m >= 0 else 0.0)
        where[q] = p1
        who[p1], who[p0] = q, m
        if m >= 0:
            where[m] = p0
        delta = pull(q, p1) + (pull(m, p0) if m >= 0 else 0.0) - before
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            cost += delta
            if cost < best - 1e-12:
                best, best_where = cost, dict(where)
        else:
            where[q] = p0
            who[p0], who[p1] = q, m
            if m >= 0:
                where[m] = p1
    lay = [-1] * L
    for q, p in best_where.items():
        lay[q] = p
    return complete_layout(chip, lay, L, rng)


def find_embedding(chip, gates, rng, node_cap=60000, deadline=None):
    """Injective logical->physical map making every gate pair adjacent (None if not found within the cap)."""
    pat = {}
    for a, b in gates:
        pat.setdefault(a, set()).add(b)
        pat.setdefault(b, set()).add(a)
    if not pat:
        return {}
    if len(pat) > chip.n or sum(len(v) for v in pat.values()) // 2 > chip.n_edges:
        return None
    if max(len(v) for v in pat.values()) > max(chip.deg):
        return None
    order, placed, rest = [], set(), set(pat)
    while rest:
        pool = [x for x in rest if pat[x] & placed] or list(rest)
        x = max(pool, key=lambda v: (len(pat[v] & placed), len(pat[v]), -v))
        order.append(x)
        placed.add(x)
        rest.discard(x)
    img, taken, budget = {}, [False] * chip.n, [node_cap]

    def go(i):
        if i == len(order):
            return True
        budget[0] -= 1
        if budget[0] <= 0 or (deadline and budget[0] % 256 == 0 and time.perf_counter() > deadline):
            raise TimeoutError
        x = order[i]
        anchors = [img[y] for y in pat[x] if y in img]
        if anchors:
            cand = set(chip.nbrset[anchors[0]])
            for s in anchors[1:]:
                cand &= chip.nbrset[s]
            cand = [c for c in cand if not taken[c]]
        else:
            cand = [c for c in range(chip.n) if not taken[c]]
        need = sum(1 for y in pat[x] if y not in img)
        cand = [c for c in cand if chip.deg[c] >= len(pat[x]) and sum(1 for z in chip.nbr[c] if not taken[z]) >= need]
        rng.shuffle(cand)
        for c in cand:
            img[x], taken[c] = c, True
            if go(i + 1):
                return True
            taken[c] = False
            del img[x]
        return False

    try:
        return dict(img) if go(0) else None
    except TimeoutError:
        return None


def prefix_layouts(chip, seq, L, rng, deadline, count=3):
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if find_embedding(chip, seq[:mid], rng, node_cap=20000, deadline=deadline):
            lo = mid
        else:
            hi = mid - 1
    out = []
    if lo == 0:
        return out
    seen = set()
    for _ in range(count * 3):
        if len(out) >= count or time.perf_counter() > deadline:
            break
        emb = find_embedding(chip, seq[:lo], rng, node_cap=20000, deadline=deadline)
        if not emb:
            continue
        key = tuple(sorted(emb.items()))
        if key in seen:
            continue
        seen.add(key)
        part = [-1] * L
        for q, p in emb.items():
            part[q] = p
        out.append(grow_layout(chip, seq, L, rng, partial=part))
    return out


# ======================================================================================================
# Exact local re-optimization ("window shave")
# ======================================================================================================
def _replay(chip, layout, moves, upto):
    """Physical position of every logical qubit after applying moves[:upto] starting from `layout`."""
    pos = list(layout)
    occ = [-1] * chip.n
    for q, p in enumerate(pos):
        occ[p] = q
    for k in range(upto):
        for x, y in moves[k]:
            qx, qy = occ[x], occ[y]
            occ[x], occ[y] = qy, qx
            if qx >= 0:
                pos[qx] = y
            if qy >= 0:
                pos[qy] = x
    return pos


def _win_bound(dist, pos, live, goal):
    """Admissible-ish lower bound: each swap can shorten two live qubits' distance to goal by 1 each."""
    return (sum(dist[pos[q]][goal[q]] for q in live) + 1) // 2


def shave_window(chip, seq, layout, moves, lo, hi, cut, deadline, node_cap=120_000):
    """Branch-and-bound: is there a routing of gates[lo:hi] using `cut` fewer swaps than `moves` currently
    does there, that leaves every qubit touched again after `hi` in exactly the same physical spot the
    original routing leaves it in?  Gates strictly before `lo` and after `hi` are untouched, so the answer
    (if any) can be spliced straight into `moves` without disturbing anything outside the window.

    Every logical qubit already has a physical slot at all times in this router (no lazy placement), so
    unlike a lazy-placement router this needs no special-casing for "not yet used" qubits.

    Returns a replacement list of per-gate swap tuples for gates[lo:hi], or None (nothing found in the
    time/node budget -- which does NOT prove no cheaper routing exists, just that this search didn't find one).
    """
    start = tuple(_replay(chip, layout, moves, lo))
    end = _replay(chip, layout, moves, hi)
    live = sorted({q for g in seq[hi:] for q in g})
    goal = {q: end[q] for q in live}
    cap = sum(len(moves[k]) for k in range(lo, hi)) - cut
    if cap < 0:
        return None
    dist, nbr = chip.dist, chip.nbr
    memo, trail, budget = {}, [], [node_cap]

    def dfs(k, pos, used):
        budget[0] -= 1
        if budget[0] <= 0 or (budget[0] % 512 == 0 and time.perf_counter() > deadline):
            raise TimeoutError
        mark = len(trail)
        while k < hi and dist[pos[seq[k][0]]][pos[seq[k][1]]] == 1:
            trail.append(("g", k))
            k += 1
        lb = _win_bound(dist, pos, live, goal)
        if k < hi:
            a, b = seq[k]
            lb = max(lb, dist[pos[a]][pos[b]] - 1)
        if used + lb > cap:
            del trail[mark:]
            return False
        if k == hi:
            if lb == 0:
                return True
            del trail[mark:]
            return False
        key = (k, pos)
        prev = memo.get(key)
        if prev is not None and prev <= used:
            del trail[mark:]
            return False
        memo[key] = used
        occ = {}
        for q, p in enumerate(pos):
            occ[p] = q
        seen, scored = set(), []
        for x in range(chip.n):
            qx = occ.get(x, -1)
            for y in nbr[x]:
                e = (x, y) if x < y else (y, x)
                if e in seen:
                    continue
                qy = occ.get(y, -1)
                if qx < 0 and qy < 0:
                    continue
                seen.add(e)
                np_ = list(pos)
                if qx >= 0:
                    np_[qx] = e[1]
                if qy >= 0:
                    np_[qy] = e[0]
                np_ = tuple(np_)
                scored.append((_win_bound(dist, np_, live, goal), np_, e))
        scored.sort(key=lambda t: t[0])
        for lbn, np_, sw in scored:
            if used + 1 + lbn > cap:
                continue
            trail.append(("s", k, sw))
            if dfs(k, np_, used + 1):
                return True
            trail.pop()
        del trail[mark:]
        return False

    try:
        ok = dfs(lo, start, 0)
    except TimeoutError:
        return None
    if not ok:
        return None
    won = {k: [] for k in range(lo, hi)}
    for step in trail:
        if step[0] == "s":
            won[step[1]].append(step[2])
    return [tuple(won[k]) for k in range(lo, hi)]


# ======================================================================================================
# Campaign driver
# ======================================================================================================
class Campaign:
    def __init__(self, program, graph, seed=0):
        self.program = [tuple(op) for op in program]
        self.graph = graph
        self.chip = Chip(graph)
        self.rng = random.Random(seed)
        self.names = sorted({q for op in self.program for q in op[1:]})
        self.lix = {q: i for i, q in enumerate(self.names)}
        self.L = len(self.names)
        self.seq = [(self.lix[o[1]], self.lix[o[2]]) for o in self.program if o[0] == "2Q"]
        self.rseq = self.seq[::-1]
        self.best_cost = math.inf
        self.best = None  # (placement, routed)
        self.best_layout = None  # list[int]: logical -> physical index, backing self.best
        self.best_moves = None  # list[tuple]: per-gate swap tuples, backing self.best
        self.floor = 0.5 * self._crit_path()
        self.pool = []  # elite layouts: [cost, layout tuple]

    def _crit_path(self):
        t = [0] * self.L
        d = 0
        for a, b in self.seq:
            x = max(t[a], t[b]) + 1
            t[a] = t[b] = x
            d = max(d, x)
        return d

    def finished(self, deadline):
        return time.perf_counter() > deadline or self.best_cost <= self.floor + 1e-9

    def _emit(self, layout, moves):
        """Turn a (layout, per-gate swap moves) pair into official (placement, routed) format."""
        chip = self.chip
        pos = list(layout)
        occ = [-1] * chip.n
        for q, p in enumerate(pos):
            occ[p] = q
        lab = chip.labels
        routed, g = [], 0
        for op in self.program:
            if op[0] == "2Q":
                for x, y in moves[g]:
                    routed.append(("SWAP", lab[x], lab[y]))
                    qx, qy = occ[x], occ[y]
                    occ[x], occ[y] = qy, qx
                    if qx >= 0:
                        pos[qx] = y
                    if qy >= 0:
                        pos[qy] = x
                routed.append(("2Q", lab[pos[self.lix[op[1]]]], lab[pos[self.lix[op[2]]]]))
                g += 1
            else:
                routed.append(("1Q", lab[pos[self.lix[op[1]]]]))
        placement = {self.names[q]: lab[p] for q, p in enumerate(layout)}
        return placement, routed

    # -- turn a search result into an official-format answer and keep it if it is better ------------
    def commit(self, layout, node):
        cost = node.nsw + 0.5 * node.dep
        if cost >= self.best_cost - 1e-9:
            return False
        moves = _moves_of(node)
        placement, routed = self._emit(layout, moves)
        if not check(self.program, self.graph, placement, routed):
            return False
        real = tally(routed)
        if real < self.best_cost - 1e-9:
            self.best_cost, self.best = real, (placement, routed)
            self.best_layout, self.best_moves = list(layout), moves
            return True
        return False

    def remember(self, cost, layout, cap=8):
        key = tuple(layout)
        if any(e[1] == key for e in self.pool):
            return
        self.pool.append([cost, key])
        self.pool.sort(key=lambda e: e[0])
        del self.pool[cap:]

    def polish(self, deadline, lens=(6, 8, 10, 12, 16), slice_time=0.5):
        """Sweep the current best solution for windows that can be re-routed with one fewer swap.
        Only ever accepts a strict, re-validated improvement, so this can never make the answer worse."""
        if self.best_moves is None:
            return
        chip, seq = self.chip, self.seq
        G = len(seq)
        moved = True
        while moved and time.perf_counter() < deadline and self.best_cost > self.floor + 1e-9:
            moved = False
            layout, moves = self.best_layout, self.best_moves
            for ln in lens:
                lo = 0
                while lo + ln <= G:
                    if time.perf_counter() > deadline or self.best_cost <= self.floor + 1e-9:
                        return
                    hi = lo + ln
                    if sum(len(moves[k]) for k in range(lo, hi)) > 0:
                        sub_deadline = min(deadline, time.perf_counter() + slice_time)
                        new = shave_window(chip, seq, layout, moves, lo, hi, 1, sub_deadline)
                        if new is not None:
                            cand = list(moves)
                            cand[lo:hi] = new
                            placement, routed = self._emit(layout, cand)
                            if check(self.program, self.graph, placement, routed):
                                real = tally(routed)
                                if real < self.best_cost - 1e-9:
                                    self.best_cost, self.best = real, (placement, routed)
                                    self.best_moves = cand
                                    moves = cand
                                    moved = True
                    lo += max(2, ln // 2)

    # -- one relaxation chain: fwd, then repeated (bwd, fwd) --------------------------------------
    def chain(self, layout, deadline, width, rounds=10, patience=2, **kw):
        slack = kw.pop("slack", 1)
        rng = self.rng
        cur_layout = list(layout)
        node = beam_pass(self.chip, self.seq, cur_layout, width, slack, deadline, rng, **kw)
        if node is None:
            return None
        self.commit(cur_layout, node)
        best = (node.nsw + 0.5 * node.dep, list(cur_layout))
        stale = 0
        for _ in range(rounds):
            if self.finished(deadline):
                break
            back = beam_pass(self.chip, self.rseq, node.pos, width, slack, deadline, rng, **kw)
            if back is None:
                break
            cur_layout = list(back.pos)
            node = beam_pass(self.chip, self.seq, cur_layout, width, slack, deadline, rng, **kw)
            if node is None:
                break
            self.commit(cur_layout, node)
            c = node.nsw + 0.5 * node.dep
            if c < best[0] - 1e-9:
                best, stale = (c, list(cur_layout)), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.remember(best[0], best[1])
        return best

    def fresh_layout(self):
        rng, chip = self.rng, self.chip
        if rng.random() < 0.35:
            return grow_layout(chip, self.seq, self.L, rng)
        lay = anneal_layout(chip, self.seq, self.L, rng, decay=rng.choice((1.0, 0.9, 0.8, 0.7)), iters=rng.choice((800, 2500)))
        return lay or grow_layout(chip, self.seq, self.L, rng)

    def run(self, deadline):
        chip, rng = self.chip, self.rng
        if self.L > chip.n:
            return
        if not self.seq:
            lay = list(range(self.L))
            self.commit(lay, _root(chip, lay))
            return
        t0 = time.perf_counter()
        span = deadline - t0
        # zero-SWAP: does the whole interaction graph embed?
        emb = find_embedding(chip, self.seq, rng, node_cap=200000, deadline=t0 + 0.2 * span)
        if emb:
            part = [-1] * self.L
            for q, p in emb.items():
                part[q] = p
            lay = complete_layout(chip, part, self.L, rng)
            node = beam_pass(chip, self.seq, lay, 1, 0, deadline, rng)
            if node is not None:
                self.commit(lay, node)
            if self.finished(deadline):
                return
        starts = prefix_layouts(chip, self.seq, self.L, rng, t0 + 0.3 * span) + [grow_layout(chip, self.seq, self.L, rng)]
        # Reserve a slice of the budget for the exact window-polish pass at the end: it can only help
        # (every candidate is re-validated), so it's worth cutting the exploration stages a bit short for.
        polish_reserve = min(0.35 * span, max(0.4, 0.08 * span))
        main_deadline = deadline - polish_reserve

        def main_done():
            return time.perf_counter() > main_deadline or self.best_cost <= self.floor + 1e-9

        # stage 1: many cheap chains
        end1 = time.perf_counter() + 0.5 * (main_deadline - time.perf_counter())
        i = 0
        while time.perf_counter() < end1 and not main_done():
            lay = starts[i] if i < len(starts) else self.fresh_layout()
            i += 1
            self.chain(lay, main_deadline, 16, noise=rng.choice((0.0, 0.0, 0.3)))
        # stage 2: wider chains on the best layouts, escalating width on the single best one
        end2 = time.perf_counter() + 0.65 * (main_deadline - time.perf_counter())
        for w in (64, 192):
            for cost, lay in list(self.pool)[:4]:
                if time.perf_counter() > end2 or main_done():
                    break
                self.chain(list(lay), main_deadline, w, rounds=6)
        if self.pool and not main_done():
            top = list(self.pool[0][1])
            for w in (512, 1536):
                if main_done():
                    break
                self.chain(top, main_deadline, w, rounds=4)
        # stage 3: noisy, re-parameterised chains around the elites until the reserved time is hit
        while not main_done():
            if not self.pool:
                self.chain(self.fresh_layout(), main_deadline, 16)
                continue
            pick = self.pool[min(len(self.pool) - 1, int(rng.random() ** 2 * len(self.pool)))]
            self.chain(
                list(pick[1]),
                main_deadline,
                rng.choice((32, 64)),
                rounds=5,
                slack=rng.choice((0, 1, 1)),
                noise=rng.choice((0.3, 0.6, 1.0)),
                look=rng.choice((10, 20)),
                decay=rng.choice((0.6, 0.8)),
            )
        # final pass: exact re-optimization of small windows in the best solution found so far
        self.polish(deadline)


# ======================================================================================================
# Public API
# ======================================================================================================
def solve(program, hardware_graph, time_budget=None, seed=0):
    """Return (initial_placement, routed_program)."""
    budget = DEFAULT_BUDGET if time_budget is None else time_budget
    try:
        camp = Campaign(program, hardware_graph, seed)
        camp.run(time.perf_counter() + budget)
        if camp.best is not None:
            return camp.best
    except Exception:
        pass
    return naive_route(program, hardware_graph)


def _run_one(args):
    program, hardware_graph, time_budget, seed = args
    placement, routed = solve(program, hardware_graph, time_budget, seed)
    return tally(routed), placement, routed


def solve_many(program, hardware_graph, time_budget=None, seeds=(0, 1, 2, 3), workers=None):
    """Best-of-N over seeds, in parallel processes when possible.  Same wall-clock cost as one run if
    the machine has >= len(seeds) free cores; falls back to running seeds one after another otherwise.
    Every candidate has already been validated inside solve(), so this just keeps the lowest score."""
    import os

    budget = DEFAULT_BUDGET if time_budget is None else time_budget
    jobs = [(program, hardware_graph, budget, s) for s in seeds]
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers > 1:
        try:
            from multiprocessing import Pool

            with Pool(workers) as pool:
                results = pool.map(_run_one, jobs)
        except Exception:
            results = [_run_one(j) for j in jobs]
    else:
        results = [_run_one(j) for j in jobs]
    best = min(results, key=lambda r: r[0])
    return best[1], best[2]