# -*- coding: utf-8 -*-
"""Remaining-lever push: UNIFIED defense = per-segment coordinator (distributed, survives partition)
+ de-jam (availability) + relay (reconnect). Test across ALL threat classes vs plain coordinator,
to lift the only weak point (connectivity A-CONN) without losing the rest. Team metric."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}


def make_red(v):
    class R(brains._Red):
        VECS = list(v)
        def get_action(self, o, a):
            if self.mem.get("target") is not None and o.get("success") is True: return self._emit(5, o)
            i = int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[i % len(self.VECS)]])), o)
    return R


def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1); return (d < ml) & (d > 0)


def components(present, A, n):
    seen = set(); out = []
    for s in present:
        if s in seen: continue
        comp = set(); st = [s]
        while st:
            u = st.pop()
            if u in seen: continue
            seen.add(u); comp.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen: st.append(v)
        out.append(comp)
    return out


def retake_nearest(env, a, i, comp, pos, ip2d, sleep):
    if not comp: return sleep
    tgt = min(comp, key=lambda c: np.linalg.norm(pos[i] - pos[c]))
    idx = actions.action_index_map(env, a)
    for j, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == tgt: return j
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep


def rollout(seed, red, mode, vectors, frag_K=0, R_relay=4):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:frag_K]) if frag_K else set()
    k = len(vectors); cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs
        comps = components(present, A, n); big = max(comps, key=len) if comps else set()
        isolated = present - big
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS); jam = red_jam > 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        # relay reconnect (ultimate only)
        reconnected = set(list(isolated)[:R_relay]) if mode == "ultimate" else set()
        iso2 = isolated - reconnected
        if mode == "ultimate":
            seg_groups = comps; unreachable = hubs              # per-segment: isolated self-defend
        else:  # plain coordinator
            seg_groups = [big | reconnected]; unreachable = hubs | iso2
        assign = {}
        for seg in seg_groups:
            sc = [c for c in seg if c in comp]; scl = [d for d in seg if d not in comp]
            used = set()
            for c in sc:
                cand = sorted([d for d in scl if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand: assign[cand[0]] = c; used.add(cand[0])
        dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign: acts[a] = retake_nearest(env, a, assign[i], comp, pos, ip2d, sleep)
            elif mode == "ultimate" and jam: dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam) if mode == "ultimate" else 0
        af.append(max(0.0, (n - len(comp) - len(hubs) - len(iso2) - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av


THREATS = [("웜", ["W"], 0), ("재밍", ["J"], 0), ("동시(W+J+B)", ["W", "J", "B"], 0),
           ("A-CONN 연결성단절", ["W", "J", "B"], 4), ("A4 SATCOM류", ["W", "J"], 3)]
print("=== UNIFIED ultimate (per-segment+de-jam+relay) vs 코디네이터 (곱셈종합) ===")
print("위협".ljust(16) + "코디네이터    ★통합(ultimate)")
co_all, ul_all = [], []
for name, vec, K in THREATS:
    red = make_red(vec)
    co = float(np.mean([rollout(s, red, "coord", vec, K) for s in EVAL]))
    ul = float(np.mean([rollout(s, red, "ultimate", vec, K) for s in EVAL]))
    co_all.append(co); ul_all.append(ul)
    print(f"{name:16}{co:.3f}        {ul:.3f}   {'↑' if ul>co+0.005 else ('=' if abs(ul-co)<=0.005 else '↓')}")
print(f"\n평균: 코디네이터 {np.mean(co_all):.3f} -> 통합 {np.mean(ul_all):.3f}")
print(f"worst-case(최악 위협): 코디네이터 {min(co_all):.3f} -> 통합 {min(ul_all):.3f}")
with open(os.path.join(OUT, "summary_ultimate.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["threat", "coordinator", "ultimate"])
    for (name, _, _), co, ul in zip(THREATS, co_all, ul_all): wr.writerow([name, round(co, 3), round(ul, 3)])
print("DONE_ULT")
