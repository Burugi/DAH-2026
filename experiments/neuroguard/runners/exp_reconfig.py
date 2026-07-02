# -*- coding: utf-8 -*-
"""동적 재구성(self-healing topology)로 특이점을 닫나? 발굴된 특이점(웜+분할K=6)에 대해,
방어가 '끊긴 노드를 매 스텝 H개씩 다시 엮는(reposition/bridge)' 동적 재구성을 수행.
H(재구성 속도)를 0→6 스윕. H↑로 곱셈이 K=0 기준(0.697)까지 회복하면 = '연결 전환로직으로 막힌다',
필요 H가 특정값 이상이면 = '재구성 속도 vs 절단 속도' 경쟁 = 처방=빠른 재구성."""
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

def rollout(seed, red, vectors, frag_K, heal_H):
    """heal_H = 매 스텝 다시 엮는 노드 수(동적 재구성 속도). 누적 bridged로 끊긴 용량 복구."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:frag_K]) if frag_K else set()
    k = len(vectors); cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs; comps = components(present, A, n)
        big = max(comps, key=len) if comps else set(); isolated = present - big
        disrupted = len(hubs) + len(isolated)            # 끊겨서 임무 이탈한 총량
        bridged = min(disrupted, int(heal_H * (t + 1)))  # 동적 재구성 누적 복구량
        eff_lost = max(0, disrupted - bridged)           # 재구성 후 남은 상실
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS); jam = red_jam > 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
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
        af.append(max(0.0, (n - len(comp) - eff_lost - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

VEC = ["W", "W", "J"]; K = 6; red = make_red(VEC)
print("=== 동적 재구성(연결 전환로직) H 스윕 — 특이점(웜+K6) 닫히나? ===", flush=True)
print("재구성속도 H   통합방어+재구성 곱셈")
rows = []
for H in [0, 1, 2, 3, 4, 6]:
    m = float(np.mean([rollout(s, red, VEC, K, H) for s in EVAL]))
    rows.append((H, round(m, 3))); print(f"  H={H:<8}{m:.3f}", flush=True)
ref = float(np.mean([rollout(s, red, VEC, 0, 0) for s in EVAL]))  # 분할없음(K=0) 상한
lo = rows[0][1]; hi = rows[-1][1]
print(f"\n(상한) 분할 K=0 기준: {ref:.3f}")
print(f"판정: H=0 {lo:.3f} -> H=6 {hi:.3f}  ({'★재구성으로 닫힘 = 연결 전환로직이 특이점을 막는다' if hi > lo + 0.15 else '재구성으로도 부분적'})")
with open(os.path.join(OUT, "summary_reconfig.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["heal_H", "곱셈"]); wr.writerows(rows); wr.writerow(["K=0_상한", round(ref, 3)])
print("DONE_RECONFIG")
