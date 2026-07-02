# -*- coding: utf-8 -*-
"""Density sensitivity of the fragmentation (A-CONN) attack. Sweep link radius (connectivity density);
count cut-vertices; measure A-CONN effectiveness vs coordinator. Hypothesis: dense/2-connected swarm
has NO cut-vertices -> fragmentation FAILS. Finds the density threshold where the ceiling vanishes."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml, csv
import run
from agents import brains, actions
base = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
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
    disc = [-1] * n; low = [0] * n; ap = set(); timer = [0]
    def dfs(u, parent):
        disc[u] = low[u] = timer[0]; timer[0] += 1; children = 0
        for v in adj[u]:
            if disc[v] == -1:
                children += 1; dfs(v, u); low[u] = min(low[u], low[v])
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


def rollout(seed, ml, K):
    cfg = dict(base); cfg["fleet"] = dict(base["fleet"]); cfg["fleet"]["max_link"] = ml
    cfg["sim"] = dict(base["sim"]); cfg["sim"]["max_length_data_links"] = ml
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red()); n = fleet["n"]
    pos0 = fleet["pos_true"][0]; adj0 = nbrs(adjacency(pos0, ml), n); deg = adjacency(pos0, ml).sum(1)
    ap = articulation_points(adj0, n)
    ranked = sorted(ap, key=lambda i: -deg[i]); jammed = set(int(x) for x in ranked[:K])
    if len(jammed) < K:
        for i in np.argsort(-deg):
            if int(i) not in jammed:
                jammed.add(int(i))
            if len(jammed) >= K:
                break
    n_ap = len(ap)
    cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; adj = nbrs(adjacency(pos, ml), n)
        present = set(range(n)) - jammed
        big = largest_comp(present, adj, n) if present else set()
        isolated = present - big; unreachable = jammed | isolated
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in unreachable]
        reachable = comp - unreachable; assign = {}; used = set()
        for c in reachable:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand:
                assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable:
                acts[a] = actions.make_blue_index(0, env, a, ctx if False else {"compromised": comp, "ip_to_drone": ip2d, "n": n}); continue
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            if i in comp:
                acts[a] = actions.make_blue_index(3, env, a, ctx)
            elif i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - len(unreachable)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()):
            break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return final, float(np.mean([1 - final, 1 - auc, COMP_F1])) * av, n_ap


def ev(ml, K=4):
    rs = [rollout(s, ml, K) for s in EVAL]
    return (float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs])), float(np.mean([r[2] for r in rs])))


print("=== density sensitivity of fragmentation attack (A-CONN K=4) ===")
print("max_link".ljust(10) + "절단점수   점령    곱셈종합")
rows = []
for ml in [26, 32, 40, 48, 60]:
    fc, m, nap = ev(ml); rows.append((ml, nap, fc, m))
    print(f"{ml:<10}{nap:6.1f}     {fc:.3f}   {m:.3f}")
with open(os.path.join(OUT, "summary_density.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["max_link", "cut_vertices", "점령", "곱셈"])
    for r in rows:
        wr.writerow([r[0], round(r[1], 1), round(r[2], 3), round(r[3], 3)])
print(f"\n희소(ml=26) 곱셈 {rows[0][3]:.3f} -> 조밀(ml=60) {rows[-1][3]:.3f}: 조밀할수록 절단점↓ 분할공격 무력화")
print("DONE_DENSITY")
