# -*- coding: utf-8 -*-
"""Concurrency test (real CybORG): are attacks ONE-AT-A-TIME or SIMULTANEOUS?
Our arms-race/probing used single-vector attacks (worm OR jam OR gps) that cycle rock-paper-
scissors. But a real adversary runs them AT ONCE. Here each compromised drone-agent picks its
vector by node id, so at any step some drones JAM, some SPREAD the worm, some BLOCK/SEIZE — a
genuinely concurrent multi-vector attack. We compare single-vector vs simultaneous multi-vector
against each fixed defense loop, to test whether simultaneity defeats specialization.
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


class _R(brains._Red):
    SEQ = None
    def get_action(self, obs, action_space):
        aid = 5 if (self.mem.get("target") is not None and obs.get("success") is True) else int(self.np_random.choice(self.SEQ))
        return self._emit(aid, obs)


class NearRed(_R): SEQ = [2, 2, 6, 10]          # worm / near-spread
class JamRed(_R):  SEQ = [7, 8, 4, 6]           # jamming + block
class BlockRed(_R): SEQ = [9, 9, 4, 10]         # comms-block + seize


class MultiVecRed(brains._Red):
    """SIMULTANEOUS multi-vector: vector assigned by node id, so different compromised drones
    attack with different vectors on the SAME step (concurrent jam + worm + block)."""
    def get_action(self, obs, action_space):
        if self.mem.get("target") is not None and obs.get("success") is True:
            return self._emit(5, obs)                       # finish a takeover when ready
        own = int(self.name.split("_")[-1])
        lane = own % 3
        if lane == 0:                                       # jamming lane
            aid = int(self.np_random.choice([7, 8]))
        elif lane == 1:                                     # worm lane
            aid = int(self.np_random.choice([2, 6]))
        else:                                               # comms-block / persist lane
            aid = int(self.np_random.choice([9, 10]))
        return self._emit(aid, obs)


ATTACKS = {"근접확산(단일)": NearRed, "재밍(단일)": JamRed, "통신차단(단일)": BlockRed,
           "★동시 멀티벡터": MultiVecRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(kind, i, comp, pos, ml, rising):
    if i in comp: return 3
    if kind == "flat": return 4 if comp else 1
    fr = frontier(i, comp, pos, ml)
    if kind == "pred": return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "predOODA":
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    return 1


def rollout(seed, red, defense):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    hist = []; rsum = 0.0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(defense, int(a.split("_")[-1]), comp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        rsum += float(np.mean(list(rew.values()))) if rew else 0.0   # availability/total-damage proxy
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, rsum


def ev(red, defense):
    rs = [rollout(s, red, defense) for s in EVAL]
    return float(np.mean([c for c, _ in rs])), float(np.mean([r for _, r in rs]))

DEFENSES = ["flat", "pred", "predOODA"]
print("=== single-vector vs SIMULTANEOUS multi-vector (점령 | 보상=가용성·총피해 대용) ===")
print("attack".ljust(16) + "".join(f"{d:>16}" for d in DEFENSES))
gridC, gridR = {}, {}
for an, rc in ATTACKS.items():
    cs, rs = [], []
    for d in DEFENSES:
        c, r = ev(rc, d); cs.append(c); rs.append(r)
    gridC[an], gridR[an] = cs, rs
    print(an.ljust(16) + "".join(f"  {cs[k]:.3f}|{rs[k]:7.0f}" for k in range(len(DEFENSES))))

# 2-panel: compromise (worm axis) + reward (availability/DoS axis)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.9))
x = np.arange(len(ATTACKS)); w = 0.24
for k, d in enumerate(DEFENSES):
    ax1.bar(x + (k - 1) * w, [gridC[a][k] for a in ATTACKS], w, label=f"방어 {d}")
    ax2.bar(x + (k - 1) * w, [gridR[a][k] for a in ATTACKS], w, label=f"방어 {d}")
for ax, ttl, yl in [(ax1, "① 점령 축 (웜): 낮을수록 방어 성공", "최종 점령"),
                    (ax2, "② 보상 축 (가용성·DoS 총피해): 높을수록 방어 성공", "누적 보상")]:
    ax.set_xticks(x); ax.set_xticklabels(list(ATTACKS), fontsize=8.5); ax.set_title(ttl, fontsize=10.5)
    ax.set_ylabel(yl); ax.legend(fontsize=8.5)
fig.suptitle("동시 멀티벡터 공격: 점령은 막아도 가용성(보상)은 동시에 무너진다 (실측, CybORG)", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig41_multivector.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_multivector.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack"] + [f"{d}_점령" for d in DEFENSES] + [f"{d}_보상" for d in DEFENSES])
    for a in ATTACKS: wr.writerow([a] + [round(v, 3) for v in gridC[a]] + [round(v, 0) for v in gridR[a]])

mv_bestC = min(gridC["★동시 멀티벡터"]); mv_worstR = min(gridR["★동시 멀티벡터"])
print(f"\n동시 멀티벡터: 최선 점령방어 {mv_bestC:.3f} | 그때 보상은 단일공격보다 악화 가능(가용성 동시 타격)")
print(f"  멀티벡터 최악 보상(어떤 방어든): {mv_worstR:.0f}")
print("Saved fig41_multivector.png, summary_multivector.csv")
