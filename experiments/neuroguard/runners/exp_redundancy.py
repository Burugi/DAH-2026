# -*- coding: utf-8 -*-
"""THE SOLUTION to the connectivity ceiling. Against fragmentation (A-CONN, articulation K=4):
  none        : centralized coordinator; isolated drones do nothing (reservoir)   [baseline]
  relay       : bridge R isolated back to the giant component (reconnect)
  local_auto  : DECENTRALISED — every connected COMPONENT runs its own local coordinator, so an
                isolated segment still self-cleans + retakes within itself (no undefended reservoir)
  combined    : relay + local_auto (bridge what you can, self-defend the rest)
This is BOTH an attack scenario (target sparse/cut-vertex topology) and a defense scenario
(maintain redundancy + per-segment autonomy). Team metric."""
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
    disc = [-1] * n; low = [0] * n; ap = set(); timer = [0]
    def dfs(u, parent):
        disc[u] = low[u] = timer[0]; timer[0] += 1; ch = 0
        for v in adj[u]:
            if disc[v] == -1:
                ch += 1; dfs(v, u); low[u] = min(low[u], low[v])
                if parent != -1 and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent:
                low[u] = min(low[u], disc[v])
        if parent == -1 and ch > 1:
            ap.add(u)
    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    return ap


def components(present, adj, n):
    seen = set(); out = []
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
        out.append(comp)
    return out


def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node:
            return i
    c = idx.get("RetakeControl", [])
    return c[0][0] if c else sleep


def pick_artic(adj0, deg, n, K):
    ap = articulation_points(adj0, n); ranked = sorted(ap, key=lambda i: -deg[i]); out = list(ranked[:K])
    for i in np.argsort(-deg):
        if len(out) >= K:
            break
        if int(i) not in out:
            out.append(int(i))
    return set(int(x) for x in out[:K])


def rollout(seed, defense, K=4, R_relay=4, ml_override=None):
    cfg2 = cfg
    if ml_override is not None:
        cfg2 = dict(cfg); cfg2["fleet"] = dict(cfg["fleet"]); cfg2["fleet"]["max_link"] = ml_override
        cfg2["sim"] = dict(cfg["sim"]); cfg2["sim"]["max_length_data_links"] = ml_override
    fleet, cyborg, env, ip2d = run.build_env(cfg2, seed, make_red()); n = fleet["n"]; ml = cfg2["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; adj0 = nbrs(adjacency(pos0, ml), n); deg = adjacency(pos0, ml).sum(1)
    jammed = pick_artic(adj0, deg, n, K)
    cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; adj = nbrs(adjacency(pos, ml), n)
        present = set(range(n)) - jammed
        comps = components(present, adj, n)
        big = max(comps, key=len) if comps else set()
        isolated = present - big
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        # relay: reconnect up to R isolated into reachable set
        reconnected = set()
        if defense in ("relay", "combined"):
            for x in list(isolated)[:R_relay]:
                reconnected.add(x)
            isolated = isolated - reconnected
        # build assignment PER component (local_auto/combined) or centralized (none/relay)
        assign = {}
        if defense in ("local_auto", "combined"):
            seg_groups = comps                                  # each component self-coordinates
        else:
            seg_groups = [big | reconnected]                    # only the giant (+reconnected) is defended
        for seg in seg_groups:
            seg_comp = [c for c in seg if c in comp]
            seg_clean = [d for d in seg if d not in comp]
            used = set()
            for c in seg_comp:
                cand = sorted([d for d in seg_clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand:
                    assign[cand[0]] = c; used.add(cand[0])
        # unreachable = jammed + (isolated drones NOT self-defending)
        if defense in ("local_auto", "combined"):
            unreachable = jammed                                # isolated still self-defend (only jammed are dead)
        else:
            unreachable = jammed | isolated
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in jammed:
                acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in unreachable:
                acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp:
                acts[a] = actions.make_blue_index(3, env, a, ctx)           # self-clean (works locally)
            elif i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - len(unreachable)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()):
            break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return final, av, float(np.mean([1 - final, 1 - auc, COMP_F1])) * av


def ev(defense, ml=None):
    rs = [rollout(s, defense, ml_override=ml) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))


allrows = []
for ml in [26, 40]:
    tag = "희소(ml=26, 분할 실재)" if ml == 26 else "기본(ml=40)"
    print(f"\n=== per-segment autonomy + redundancy vs A-CONN — {tag} ===")
    print("defense".ljust(14) + "점령    가용성   곱셈종합")
    for d in ["none", "relay", "local_auto", "combined"]:
        fc, av, m = ev(d, ml); allrows.append((ml, d, fc, av, m))
        print(f"{d:14}{fc:.3f}   {av:.3f}   {m:.3f}")
with open(os.path.join(OUT, "summary_redundancy.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["max_link", "defense", "점령", "가용성", "곱셈"])
    for r in allrows:
        wr.writerow([r[0], r[1], round(r[2], 3), round(r[3], 3), round(r[4], 3)])
print("\nDONE_REDUNDANCY")
