# -*- coding: utf-8 -*-
"""Adversarial probing: red-team our own 'optimal' predictive defense to FIND its cracks.
We craft attacker strategies that target the predictive defense's assumptions (it defends the
1-hop frontier): leap-frog (exploit FARTHEST, beyond the frontier), random, jam-heavy,
seize-rush, mixed. Evaluate each vs {flat, predictive, predOODA} to see which attacks break it.
This is defensive red-teaming to harden the design.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [3000, 3001, 3002, 3003, 3004]

# red action ids: 1 Discover,2 ExpNearest,3 ExpRandom,4 ExpFarthest,5 Seize,6 SpreadWorm,7 JamNear,8 JamFar,10 Persist
class _R(brains._Red):
    SEQ = None
    def get_action(self, obs, action_space):
        if self.mem.get("target") is not None and obs.get("success") is True:
            aid = 5                                       # finish takeover when ready
        else:
            aid = int(self.np_random.choice(self.SEQ))
        return self._emit(aid, obs)

class NearRed(_R): SEQ = [2, 2, 6, 10]                    # nearest-spread (what defense expects)
class FarRed(_R):  SEQ = [4, 4, 4, 10]                    # LEAP-FROG: exploit farthest (beyond frontier)
class RandRed(_R): SEQ = [3, 3, 4, 10]                    # random+far (unpredictable targets)
class JamRed(_R):  SEQ = [7, 8, 4, 6]                     # jam + exploit (blind + spread)
class RushRed(_R): SEQ = [2, 4, 6, 10, 10]               # aggressive multi-exploit + persist
class MixRed(_R):  SEQ = [1, 4, 3, 6, 10, 7]             # mixed creative

REDS = {"근접확산(예상)": NearRed, "★도약(최원거리)": FarRed, "랜덤+원거리": RandRed,
        "재밍+확산": JamRed, "장악러시": RushRed, "혼합창의": MixRed,
        "rule(기준)": brains.RuleRed, "rl(학습)": brains.RLRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)

def blue_act(kind, i, comp, pos, ml, rising):
    if i in comp: return 3
    if kind == "flat": return 4 if comp else 1
    fr = frontier(i, comp, pos, ml)
    if kind == "pred":
        return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "predOODA":
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    return 1

try:
    from agents.rl import QTable, RED_Q
    brains.use_rl(None, QTable.load(RED_Q))
except Exception:
    pass


def evalu(red_cls, blue_kind):
    cs = []
    for s in EVAL:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, red_cls); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_act(blue_kind, int(a.split("_")[-1]), comp, pos, ml, rising), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


BLUES = ["flat", "pred", "predOODA"]
print("=== adversarial probe: attack strategy vs defense (final compromise) ===")
print("attack".ljust(18) + "".join(f"{b:>12}" for b in BLUES))
M = {}; rows = []
for rn, rc in REDS.items():
    M[rn] = [evalu(rc, b) for b in BLUES]
    rows.append([rn] + [round(v, 3) for v in M[rn]])
    print(rn.ljust(18) + "".join(f"{v:12.3f}" for v in M[rn]))

# which attack breaks the predictive defense most?
pred_idx = BLUES.index("pred")
worst = max(M, key=lambda k: M[k][pred_idx])
print(f"\n*** 예측 방어를 가장 많이 뚫는 공격: {worst} = {M[worst][pred_idx]:.3f} (기준 근접확산 {M['근접확산(예상)'][pred_idx]:.3f})")

reds = list(REDS.keys()); x = np.arange(len(reds)); w = 0.26
plt.figure(figsize=(12, 5))
for j, b in enumerate(BLUES):
    plt.bar(x + (j - 1) * w, [M[r][j] for r in reds], w, label=f"방어={b}")
plt.xticks(x, reds, fontsize=8, rotation=15); plt.ylabel("최종 점령 (높을수록 공격 성공)")
plt.title("적대 프로빙: 공격 전략 × 방어 — 예측 방어의 틈 찾기")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig31_attack_probe.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_attack_probe.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack"] + BLUES); wr.writerows(rows)
print("Saved fig31_attack_probe.png, summary_attack_probe.csv")
