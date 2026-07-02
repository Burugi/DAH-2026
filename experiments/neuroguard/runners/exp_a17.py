# -*- coding: utf-8 -*-
"""[반례 규명] A17에서 통합방어(ultimate)가 코디네이터보다 미세 약함(0.702<... wait 0.692). 왜?
가설: 가벼운 분할(frag_K 작음)에선 ultimate의 per-segment·relay 오버헤드가 손해. frag_K를 0~6 스윕해
coord vs ultimate 교차점을 찾음 → 적응 규칙(ultimate는 frag_K≥임계에서만 켜기) 도출. 공격=웜."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml
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
        cc = set(); st = [s]
        while st:
            u = st.pop()
            if u in seen: continue
            seen.add(u); cc.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen: st.append(v)
        out.append(cc)
    return out

def retake_nearest(env, a, i, comp, pos, ip2d, sleep):
    if not comp: return sleep
    tgt = min(comp, key=lambda c: np.linalg.norm(pos[i] - pos[c]))
    idx = actions.action_index_map(env, a)
    for j, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == tgt: return j
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, red, vectors, frag_K, mode, R_relay=4):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:frag_K]) if frag_K else set()
    k = len(vectors); cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs; comps = components(present, A, n)
        big = max(comps, key=len) if comps else set(); isolated = present - big
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS); jam = red_jam > 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        reconnected = set(list(isolated)[:R_relay]) if mode == "ultimate" else set()
        iso2 = isolated - reconnected
        seg_groups = comps if mode == "ultimate" else [big | reconnected]
        unreachable = hubs if mode == "ultimate" else (hubs | iso2)
        assign = {}
        for seg in seg_groups:
            sc = [c for c in seg if c in comp]; scl = [d for d in seg if d not in comp]; used = set()
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
        lost = (len(hubs) + len(iso2)) if mode != "ultimate" else len(hubs)
        af.append(max(0.0, (n - len(comp) - lost - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

VEC = ["W"]; red = make_red(VEC)
print("=== A17 반례 규명: 분할 K별 코디네이터 vs 통합방어 (공격=웜, 곱셈종합) ===", flush=True)
print("분할 K    코디네이터   통합방어   우세")
rows = []
for K in [0, 1, 2, 3, 4, 6]:
    co = float(np.mean([rollout(s, red, VEC, K, "coord") for s in EVAL]))
    ul = float(np.mean([rollout(s, red, VEC, K, "ultimate") for s in EVAL]))
    win = "통합" if ul > co + 0.005 else ("코디" if co > ul + 0.005 else "동")
    rows.append((K, round(co, 3), round(ul, 3), win))
    print(f"  K={K:<7}{co:.3f}       {ul:.3f}     {win}", flush=True)
cross = next((K for K, co, ul, w in rows if w == "통합"), None)
print(f"\n교차점: 통합방어가 이기기 시작하는 분할 K = {cross} → 적응규칙: frag_K>={cross}에서만 통합방어 ON, 미만은 코디네이터")
with open(os.path.join(OUT, "summary_a17.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["frag_K", "코디네이터", "통합방어", "우세"]); wr.writerows(rows)
print("DONE_A17")
