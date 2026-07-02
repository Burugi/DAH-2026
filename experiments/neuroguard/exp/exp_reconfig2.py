# -*- coding: utf-8 -*-
"""동적 재구성 '실제 실측' — 자원 경쟁 모델: 재구성에 드론 M대를 쓰면 그 M대는 재장악에서 빠진다.
(끊긴 형상 복구 ↑ vs 웜 재장악 ↓ 의 트레이드오프). M을 스윕해 곱셈-최적 재구성 예산과,
비용을 물고도 특이점(웜+K6, 무재구성 0.23)을 넘는지 실측. exp_reconfig(추상)의 현실판."""
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

def rollout(seed, red, vectors, frag_K, M):
    """M = 재구성에 배정하는 클린 드론 수(재장악 풀에서 빠짐). cum_bridged = 누적 형상복구."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:frag_K]) if frag_K else set()
    k = len(vectors); cf, af = [], []; cum_bridged = 0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs; comps = components(present, A, n)
        big = max(comps, key=len) if comps else set(); isolated = present - big
        disrupted = len(hubs) + len(isolated)
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS); jam = red_jam > 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [a for a in live if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in hubs]
        # 재구성 워커 M대 (재장악 풀에서 제외) — 누적 형상 복구
        reconfig = set(clean[:M]); bridged_step = len(reconfig)
        cum_bridged = min(disrupted, cum_bridged + bridged_step) if disrupted else cum_bridged
        eff_lost = max(0, disrupted - cum_bridged)
        retake_pool = [a for a in clean if a not in reconfig]
        # coordinator retake 배정 (축소된 풀)
        assign = {}; used = set()
        for c in [x for x in comp]:
            cand = sorted([a for a in retake_pool if a not in used], key=lambda a: np.linalg.norm(pos[int(a.split("_")[-1])] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in hubs: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if a in reconfig: acts[a] = actions.make_blue_index(6, env, a, ctx); continue  # AllowTraffic/재연결
            if a in assign: acts[a] = retake_nearest(env, a, assign[a], comp, pos, ip2d, sleep)
            elif jam: dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam)
        af.append(max(0.0, (n - len(comp) - eff_lost - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av, final

VEC = ["W", "W", "J"]; K = 6; red = make_red(VEC)
print("=== 동적 재구성 실제 실측: 재구성 예산 M 트레이드오프 (웜+K6) ===", flush=True)
print("재구성예산 M   곱셈    점령")
rows = []
for M in [0, 1, 2, 3, 4, 6, 8]:
    rs = [rollout(s, red, VEC, K, M) for s in EVAL]
    mm = float(np.mean([r[0] for r in rs])); cc = float(np.mean([r[1] for r in rs]))
    rows.append((M, round(mm, 3), round(cc, 3))); print(f"  M={M:<8}{mm:.3f}   {cc:.3f}", flush=True)
best = max(rows, key=lambda r: r[1])
print(f"\n최적 재구성 예산 M={best[0]}: 곱셈 {best[1]} (무재구성 M=0 대비 {rows[0][1]}->{best[1]})")
print(f"판정: {'★비용 물어도 특이점 넘음 = 동적 재구성 실효' if best[1] > rows[0][1] + 0.15 else '트레이드오프로 제한적'}")
with open(os.path.join(OUT, "summary_reconfig2.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["reconfig_M", "곱셈", "점령"]); wr.writerows(rows)
print("DONE_RECONFIG2")
