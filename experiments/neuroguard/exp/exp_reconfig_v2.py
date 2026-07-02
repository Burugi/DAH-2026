# -*- coding: utf-8 -*-
"""통합 방어 v2 완성 실측: 동적 재구성에 ①적응형 예산(재장악 여력 확보 후 남는 드론만 재구성 → 과투자 역효과 제거)
②이동 비용(재구성 드론은 move_penalty 만큼 임무 이탈)을 넣고, 분할 강도 K(2/4/6/8) 스윕으로 강건성 실측.
비교: 무재구성 / 고정 M=6(무비용) / 적응형+비용(v2). v2가 전 K에서 특이점을 강건히 닫으면 완성."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}; MOVE_PEN = 0.5

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

def rollout(seed, red, vectors, frag_K, mode):
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
        need = max(0, disrupted - cum_bridged)
        if mode == "none": M = 0
        elif mode == "fixed6": M = min(6, len(clean))
        else:  # adaptive: 재장악 여력(=len(comp)) 확보 후 남는 드론만, need 만큼
            M = int(min(need, max(0, len(clean) - len(comp))))
        reconfig = set(clean[:M]); cum_bridged = min(disrupted, cum_bridged + len(reconfig)) if disrupted else cum_bridged
        eff_lost = max(0, disrupted - cum_bridged)
        retake_pool = [a for a in clean if a not in reconfig]
        assign = {}; used = set()
        for c in list(comp):
            cand = sorted([a for a in retake_pool if a not in used], key=lambda a: np.linalg.norm(pos[int(a.split("_")[-1])] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in hubs: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if a in reconfig: acts[a] = actions.make_blue_index(6, env, a, ctx); continue
            if a in assign: acts[a] = retake_nearest(env, a, assign[a], comp, pos, ip2d, sleep)
            elif jam: dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam)
        move_cost = MOVE_PEN * len(reconfig) if mode != "fixed6" else 0.0   # v2만 이동비용 부과
        af.append(max(0.0, (n - len(comp) - eff_lost - (red_jam - restored) - move_cost) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

VEC = ["W", "W", "J"]; red = make_red(VEC)
print(f"=== 통합방어 v2 (적응형 예산 + 이동비용 {MOVE_PEN}) — 분할 K 스윕 (웜+재밍) ===", flush=True)
print("분할 K    무재구성   고정M=6    v2(적응+비용)")
rows = []
for K in [2, 4, 6, 8]:
    none = float(np.mean([rollout(s, red, VEC, K, "none") for s in EVAL]))
    fix = float(np.mean([rollout(s, red, VEC, K, "fixed6") for s in EVAL]))
    v2 = float(np.mean([rollout(s, red, VEC, K, "adaptive") for s in EVAL]))
    rows.append((K, round(none, 3), round(fix, 3), round(v2, 3)))
    print(f"  K={K:<7}{none:.3f}      {fix:.3f}      {v2:.3f}", flush=True)
print(f"\n요약: v2가 전 K에서 무재구성 대비 {'강건히 상승' if all(r[3] > r[1] + 0.1 for r in rows) else '부분 상승'}")
with open(os.path.join(OUT, "summary_reconfig_v2.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["frag_K", "무재구성", "고정M6", "v2_적응+비용"]); wr.writerows(rows)
print("DONE_V2")
