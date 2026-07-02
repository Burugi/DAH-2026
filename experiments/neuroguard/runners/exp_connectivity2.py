# -*- coding: utf-8 -*-
"""Connectivity frontier — smarter ATTACK vs proactive DEFENSE.
ATTACK targeting : 'degree' (top-K degree hubs) vs 'artic' (true articulation points / cut-vertices).
DEFENSE          : none / reactive_relay (reconnect isolated AFTER partition) /
                   proactive (pre-guard cut-vertices so jamming them does NOT partition).
Sweep K. Team metric (comp_F1=0.866)."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866


def make_red():
    AID = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}
    class R(brains._Red):
        VECS = ["W", "J", "B"]
        def get_action(self, obs, asp):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            o = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(AID[self.VECS[o % 3]])), obs)
    return R


def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return (d < ml) & (d > 0)


def nbrs(A, n):
    return [set(np.where(A[i])[0].tolist()) for i in range(n)]


def articulation_points(adj, n):
    """Tarjan: return set of cut-vertices. adj = list of neighbor sets."""
    disc = [-1] * n; low = [0] * n; ap = set(); timer = [0]
    def dfs(u, parent):
        disc[u] = low[u] = timer[0]; timer[0] += 1; children = 0
        for v in adj[u]:
            if disc[v] == -1:
                children += 1; dfs(v, u)
                low[u] = min(low[u], low[v])
                if parent != -1 and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent:
                low[u] = min(low[u], disc[v])
        if parent == -1 and children > 1:
            ap.add(u)
    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    return ap


def largest_comp(present, adj, n):
    seen = set(); best = set()
    for s in present:
        if s in seen:
            continue
        comp = set(); st = [s]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u); comp.add(u)
            for v in adj[u]:
                if v in present and v not in seen:
                    st.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node:
            return i
    c = idx.get("RetakeControl", [])
    return c[0][0] if c else sleep


def pick_targets(mode, adj, deg, n, K):
    if mode == "degree":
        return set(int(x) for x in np.argsort(-deg)[:K])
    # articulation: prefer cut-vertices (break ties by degree)
    ap = articulation_points(adj, n)
    ranked = sorted(ap, key=lambda i: -deg[i])
    out = list(ranked[:K])
    if len(out) < K:  # pad with top-degree non-AP
        for i in np.argsort(-deg):
            if int(i) not in out:
                out.append(int(i))
            if len(out) >= K:
                break
    return set(int(x) for x in out[:K])


def rollout(seed, atk_mode, defense, K, R_budget):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red()); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; A0 = adjacency(pos0, ml); adj0 = nbrs(A0, n); deg = A0.sum(1)
    jammed = pick_targets(atk_mode, adj0, deg, n, K)
    ap0 = articulation_points(adj0, n)
    guarded = set(sorted((jammed & ap0) | ap0, key=lambda i: -deg[i])[:R_budget]) if defense == "proactive" else set()
    cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; adj = nbrs(adjacency(pos, ml), n)
        down = jammed - guarded                      # proactive: guarded cut-vertices stay up
        present = set(range(n)) - down
        big = largest_comp(present, adj, n) if present else set()
        isolated = present - big
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in down and int(a.split("_")[-1]) not in isolated]
        relays = set()
        if defense == "reactive_relay":
            for x in list(isolated)[:R_budget]:
                cand = [d for d in clean if d not in relays]
                if not cand:
                    break
                relays.add(min(cand, key=lambda d: np.linalg.norm(pos[d] - pos[x]))); isolated.discard(x)
        n_guard = min(len(guarded), len([1 for _ in guarded]))   # guards consume drones (cost)
        defenders = [d for d in clean if d not in relays][:max(0, len(clean) - len(relays) - len(guarded))]
        unreachable = down | isolated; reachable = comp - unreachable
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        assign = {}; used = set()
        for c in reachable:
            cand = sorted([d for d in defenders if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand:
                assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable:
                acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp:
                acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - len(unreachable) - len(guarded)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()):
            break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return final, av, float(np.mean([1 - final, 1 - auc, COMP_F1])) * av


def ev(atk, defense, K, R=4):
    rs = [rollout(s, atk, defense, K, R) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))


print("=== connectivity frontier: 공격 표적(degree vs articulation) x 방어 (곱셈종합) ===")
DEFS = ["none", "reactive_relay", "proactive"]
for K in [2, 4, 6]:
    print(f"\n[K={K} 노드 재밍]")
    for atk in ["degree", "artic"]:
        line = f"  공격={atk:8}"
        for d in DEFS:
            fc, av, m = ev(atk, d, K)
            line += f" | {d}:{m:.3f}(점령{fc:.2f})"
        print(line)
# headline at K=4
rows = []
for atk in ["degree", "artic"]:
    for d in DEFS:
        fc, av, m = ev(atk, d, 4); rows.append((atk, d, fc, m))
with open(os.path.join(OUT, "summary_connectivity2.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["atk_target", "defense", "점령", "곱셈"])
    for r in rows:
        wr.writerow([r[0], r[1], round(r[2], 3), round(r[3], 3)])
print("\nDONE_CONN2")
