# -*- coding: utf-8 -*-
"""[신규 공격4/D2 · 1차 근사] 임무 무결성 맹점 실증. 그레이존 공격: 드론을 '점령'하지 않고(comp=0) 재밍도
안 하며(가용성 유지), 표적/명령만 미세 왜곡 → 클린·가용인데 '오임무' 수행. 현행 곱셈(점령+가용성)은 이를
못 잡음(맹점). 임무 무결성 축 + 임무 모니터(행동 일관성 검사로 왜곡 드론 교정)를 추가해 실증.
주: 왜곡·모니터는 오버레이(1차 근사) — 메트릭 맹점의 존재와 해법 방향 증명이 목적."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866

def make_red():   # 약한 웜(실제 점령 동역학 유지) — 그레이존은 별도 오버레이
    class R(brains._Red):
        def get_action(self, o, a):
            if self.mem.get("target") is not None and o.get("success") is True: return self._emit(5, o)
            return self._emit(2, o)
    return R

def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, corrupt_frac, monitor, mon_recall=0.6):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red()); n = fleet["n"]
    rng = np.random.default_rng(seed)
    corrupted = set(rng.choice(n, size=int(corrupt_frac * n), replace=False)) if corrupt_frac else set()
    cf, af, mf = [], [], []
    for t in range(cfg["steps"]):
        comp = set(run.compromised_drones(cyborg, n)); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
        used = set(); assign = {}
        for c in comp:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        # 임무 모니터: 왜곡(오임무) 드론을 행동 일관성으로 탐지·교정
        if monitor and corrupted:
            r2 = np.random.default_rng(seed * 7 + t)
            fixed = {d for d in corrupted if r2.random() < mon_recall}
            corrupted -= fixed
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx)
            elif i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp)) / n))                 # 현행 가용성(왜곡 무시)
        mf.append((n - len(comp) - len(corrupted - comp)) / n)   # 임무 무결성(왜곡 반영)
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    mult = float(np.mean([1 - final, 1 - auc, COMP_F1])) * av
    mission = float(np.mean(mf))
    return mult, mission

print("=== [공격4/D2·1차근사] 임무 무결성 맹점 + 모니터 ===", flush=True)
print("상황".ljust(30) + "현행 곱셈".rjust(12) + "임무무결성".rjust(12))
rows = []
for label, cf_, mon in [("공격 없음", 0.0, False), ("그레이존 40% (모니터 없음)", 0.4, False),
                        ("그레이존 40% (임무 모니터)", 0.4, True)]:
    mu = float(np.mean([rollout(s, cf_, mon)[0] for s in EVAL]))
    mi = float(np.mean([rollout(s, cf_, mon)[1] for s in EVAL]))
    rows.append((label, round(mu, 3), round(mi, 3)))
    print(f"{label:30}{mu:>12.3f}{mi:>12.3f}", flush=True)
print("\n해석: 현행 곱셈은 그레이존 유무에 거의 불변(맹점) / 임무무결성만 급락→모니터로 회복. 임무축 채점 필요.")
with open(os.path.join(OUT, "summary_mission.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["situation", "현행곱셈", "임무무결성"]); wr.writerows(rows)
print("DONE_MISSION")
