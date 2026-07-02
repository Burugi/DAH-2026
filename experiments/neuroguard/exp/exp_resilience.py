# -*- coding: utf-8 -*-
"""[신규 D4/D5 + 공격3/6] 회복탄력 방어 실측.
(A) 재장악 스래싱(Persist 재감염) vs 배정정책: deterministic(매스텝 재계산) vs committed(대상 고정 유지).
(B) de-jam 고갈 페인트(jam+worm) vs 예비: no-reserve(잔여 전부 dejam) vs reserve(R대는 재장악 전용 확보).
채점=true comp 곱셈종합."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10], "P": [10, 5]}; JAM_VECS = {"J", "B"}  # P=Persist재장악

def make_red(v):
    class R(brains._Red):
        VECS = list(v)
        def get_action(self, o, a):
            if self.mem.get("target") is not None and o.get("success") is True: return self._emit(5, o)
            i = int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[i % len(self.VECS)]])), o)
    return R

def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, red, vectors, assign_mode="det", reserve=0):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]
    cf, af = [], []; k = len(vectors); commit = {}
    for t in range(cfg["steps"]):
        comp = set(run.compromised_drones(cyborg, n)); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
        # 배정
        assign = {}
        if assign_mode == "committed":
            commit = {d: c for d, c in commit.items() if c in comp and d in clean}  # 대상 아직 감염·본인 클린이면 유지
            used = set(commit)
            for d, c in commit.items(): assign[d] = c
            for c in comp:
                if c in assign.values(): continue
                cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand: assign[cand[0]] = c; used.add(cand[0]); commit[cand[0]] = c
        else:
            used = set()
            for c in comp:
                cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand: assign[cand[0]] = c; used.add(cand[0])
        free = [d for d in clean if d not in assign]           # 재장악 안 하는 클린
        reserve_set = set(free[:reserve])                       # 예비(재장악 전용, dejam 금지)
        acts = {}; dejam = 0
        for a in live:
            i = int(a.split("_")[-1])
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep); continue
            if i in reserve_set: acts[a] = actions.make_blue_index(0, env, a, ctx); continue  # 예비 대기
            if red_jam > 0: dejam += 1
            acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - red_jam + min(dejam, red_jam)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

print("=== [D4] 재장악 스래싱(Persist) vs 배정정책 ===", flush=True)
redP = make_red(["W", "P"])
det = float(np.mean([rollout(s, redP, ["W", "P"], "det") for s in EVAL]))
com = float(np.mean([rollout(s, redP, ["W", "P"], "committed") for s in EVAL]))
print(f"  deterministic 배정 {det:.3f} | committed(고정) 배정 {com:.3f} | 개선 {com-det:+.3f}", flush=True)

print("\n=== [D5] de-jam 고갈(jam+worm) vs 예비제대 R ===", flush=True)
redJW = make_red(["W", "J", "J"])
rows = [("스래싱", det, com, "")]
print("  예비 R    곱셈")
for R in [0, 2, 4, 6]:
    m = float(np.mean([rollout(s, redJW, ["W", "J", "J"], "det", reserve=R) for s in EVAL]))
    rows.append((f"reserve{R}", round(m, 3), "", "")); print(f"  R={R:<7}{m:.3f}", flush=True)
with open(os.path.join(OUT, "summary_resilience.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["case", "a", "b", "_"]); wr.writerow(["thrash_det/committed", round(det, 3), round(com, 3), ""])
    for r in rows[1:]: wr.writerow(r)
print("DONE_RESIL")
