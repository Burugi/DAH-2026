# -*- coding: utf-8 -*-
"""특이점 판별: 발굴된 신종(웜+분할K=6)이 '새 메커니즘'인가, 아니면 '방어 복원예산(relay R) 부족'인가?
통합방어의 relay 예산 R을 4→6→8→12로 늘리며 K=6 공격 곱셈을 측정. R↑로 닫히면 = 새 메커니즘 아님,
'분할K vs relayR' 용량 경쟁(=redundancy 예산 문제). 닫히면 처방=예산↑(2-연결성)."""
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

def rollout(seed, red, vectors, frag_K, R_relay):
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
        reconnected = set(list(isolated)[:R_relay]); iso2 = isolated - reconnected
        seg_groups = comps; unreachable = hubs
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
            elif jam: dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam)
        af.append(max(0.0, (n - len(comp) - len(hubs) - len(iso2) - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

# 발굴된 특이점 공격 = 웜(주) + 재밍 + 분할 K=6
VEC = ["W", "W", "W", "J"]; K = 6
red = make_red(VEC)
print("=== 특이점 판별: relay 예산 R 스윕 (공격=웜+재밍, 분할 K=6) ===", flush=True)
print("relay R    통합방어 곱셈")
rows = []
for R in [4, 6, 8, 12, 18]:
    m = float(np.mean([rollout(s, red, VEC, K, R) for s in EVAL]))
    rows.append((R, round(m, 3))); print(f"  R={R:<7}{m:.3f}", flush=True)
lo = rows[0][1]; hi = rows[-1][1]
print(f"\nR=4 {lo:.3f} -> R=18 {hi:.3f}  ({'★예산↑로 닫힘 = 새 메커니즘 아님, 용량(분할K vs relayR) 경쟁' if hi > lo + 0.1 else '예산으로 안 닫힘 = 구조적 한계'})")
with open(os.path.join(OUT, "summary_budget.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["relay_R", "ultimate_곱셈"]); wr.writerows(rows)
print("DONE_BUDGET")
