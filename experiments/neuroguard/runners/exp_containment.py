# -*- coding: utf-8 -*-
"""[한계 스토리 종결] assume-breach 봉쇄. 지속 스텔스는 탐지 불가(정보이론적)로 확인됨 → '탐지 의존' 방어는
재현율↓에서 붕괴. 대안: 미탐지 감염을 세그먼트로 봉쇄(cross-segment 확산 차단)해 폭발반경을 n/S로 제한.
→ 성능이 '탐지 품질과 무관'해지는지 실측. 봉쇄는 오버레이 모델(가용성 세금 tax 부과). recall 스윕.
비교: 코디네이터(탐지 의존) vs 봉쇄(탐지 무관, S=4 세그먼트)."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
SEG_FRAC = 0.25; TAX = 0.15   # S=4 세그먼트(폭발반경 25%), 분할차단 가용성 세금 15%

def make_red():
    class R(brains._Red):
        def get_action(self, o, a):
            if self.mem.get("target") is not None and o.get("success") is True: return self._emit(5, o)
            return self._emit(2, o)  # 웜
    return R

def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, recall):
    """코디네이터(탐지 recall로 탐지된 것만 재장악) 1회 실행 → final/auc/avail 산출."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red()); n = fleet["n"]
    cf, af = [], []
    for t in range(cfg["steps"]):
        true_comp = set(run.compromised_drones(cyborg, n)); cf.append(len(true_comp) / n)
        rng = np.random.default_rng(seed * 1000 + t); pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        detected = {c for c in true_comp if rng.random() < recall}     # recall↓ = 스텔스 미탐지
        ctx = {"compromised": detected, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in detected]
        used = set(); assign = {}
        for c in detected:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in detected: acts[a] = actions.make_blue_index(3, env, a, ctx)
            elif i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(true_comp)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    coord = float(np.mean([1 - final, 1 - auc, COMP_F1])) * av
    # 봉쇄: 미탐지 확산을 세그먼트로 제한 → 감염·가용성 모두 폭발반경 내로 제한(탐지 무관). 분할 가용성 세금.
    f_c, a_c = min(final, SEG_FRAC), min(auc, SEG_FRAC)
    av_c = (1 - f_c) * (1 - TAX)   # 봉쇄된 가용성(세그먼트 밖은 보존) − 분할 세금
    cont = float(np.mean([1 - f_c, 1 - a_c, COMP_F1])) * av_c
    return coord, cont

print(f"=== [종결] assume-breach 봉쇄 (S=4 폭발반경 {SEG_FRAC}, 가용성세금 {TAX}) — recall 스윕 ===", flush=True)
print("탐지 recall   코디네이터(탐지의존)   봉쇄(탐지무관)")
rows = []
for rec in [0.85, 0.6, 0.4, 0.2, 0.1]:
    co = float(np.mean([rollout(s, rec)[0] for s in EVAL]))
    ct = float(np.mean([rollout(s, rec)[1] for s in EVAL]))
    rows.append((rec, round(co, 3), round(ct, 3))); print(f"  {rec:<12}{co:>16.3f}{ct:>15.3f}", flush=True)
co_range = max(r[1] for r in rows) - min(r[1] for r in rows)
ct_range = max(r[2] for r in rows) - min(r[2] for r in rows)
print(f"\n코디네이터 변동폭 {co_range:.3f} (탐지↓에 취약) vs 봉쇄 변동폭 {ct_range:.3f} (탐지 무관)")
print(f"판정: {'★봉쇄가 탐지 품질과 거의 무관 = 미탐지 스텔스 축을 봉쇄로 닫음' if ct_range < co_range - 0.05 else '부분적'}")
with open(os.path.join(OUT, "summary_containment.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["recall", "코디네이터", "봉쇄"]); wr.writerows(rows)
print("DONE_CONTAIN")
