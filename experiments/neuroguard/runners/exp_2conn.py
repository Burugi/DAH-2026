# -*- coding: utf-8 -*-
"""대칭 완성: 발굴된 특이점 공격(웜+재밍, 분할 K=6)을 '구조(링크밀도↑=2-연결성)'로 막나?
반응형(통합방어)은 못 닫음(exp_budget). 여기선 링크반경 ml을 키워(밀도↑) 절단점을 없애면
같은 공격이 닫히는지 측정. sparse→dense로 곱셈 회복하면 = '특이점은 구조로만 넘는다' 증명."""
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

def rollout(seed, red, vectors, frag_K, ml, R_relay=4):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]
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
        assign = {}
        for seg in comps:
            sc = [c for c in seg if c in comp]; scl = [d for d in seg if d not in comp]; used = set()
            for c in sc:
                cand = sorted([d for d in scl if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand: assign[cand[0]] = c; used.add(cand[0])
        dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in hubs: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign: acts[a] = retake_nearest(env, a, assign[i], comp, pos, ip2d, sleep)
            elif jam: dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam)
        af.append(max(0.0, (n - len(comp) - len(hubs) - len(iso2) - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    frag = 0
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

VEC = ["W", "W", "J"]; K = 6; red = make_red(VEC)
ML0 = cfg["fleet"].get("max_link", 40)
print(f"=== 특이점 공격(웜+재밍,K=6)을 구조(링크밀도 ml↑=2-연결성)로 막나? 기본 ml={ML0} ===", flush=True)
print("링크반경 ml   통합방어 곱셈   (밀도↑ = 절단점↓ = 2-연결성)")
rows = []
for ml in [ML0, 70, 100, 150, 220]:
    m = float(np.mean([rollout(s, red, VEC, K, ml) for s in EVAL]))
    rows.append((ml, round(m, 3))); print(f"  ml={ml:<8}{m:.3f}", flush=True)
ref = float(np.mean([rollout(s, red, VEC, 0, ML0) for s in EVAL]))  # 분할 없음 기준
print(f"\n(참고) 분할 K=0 기준(공격만, 분할없음): {ref:.3f}")
lo = rows[0][1]; hi = rows[-1][1]
print(f"판정: sparse ml={ML0} {lo:.3f} -> dense ml=220 {hi:.3f}  ({'★구조로 닫힘 = 특이점은 2-연결성으로만 넘는다' if hi > lo + 0.1 else '구조로도 제한적'})")
with open(os.path.join(OUT, "summary_2conn.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["max_link_ml", "ultimate_곱셈"]); wr.writerows(rows); wr.writerow(["K=0_ref", round(ref, 3)])
print("DONE_2CONN")
