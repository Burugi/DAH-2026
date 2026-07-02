# -*- coding: utf-8 -*-
"""Constructive proof that SENSING SPEED is the lever for adaptive defense.
We grant the defender PERFECT attack identification but DELAYED by D steps: it runs the neutral
default (pred) for D steps, then switches to the oracle-best loop for the true attack. Sweeping D
shows: D=0 (instant sensing) -> oracle (best possible); each step of sensing delay costs
performance (worst-case rises). => invest in fast attack-effect sensing (B20 L1).
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
LOOPS = ["flat", "pred", "predOODA"]
DELAYS = [0, 2, 4, 6, 8, 10, 14]


class _R(brains._Red):
    SEQ = None
    def get_action(self, obs, action_space):
        aid = 5 if (self.mem.get("target") is not None and obs.get("success") is True) else int(self.np_random.choice(self.SEQ))
        return self._emit(aid, obs)
class NearRed(_R): SEQ = [2, 2, 6, 10]
class FarRed(_R):  SEQ = [4, 4, 4, 10]
class RandRed(_R): SEQ = [3, 3, 4, 10]
class JamRed(_R):  SEQ = [7, 8, 4, 6]
class RushRed(_R): SEQ = [2, 4, 6, 10, 10]
class MixRed(_R):  SEQ = [1, 4, 3, 6, 10, 7]
REDS = {"근접": NearRed, "도약": FarRed, "랜덤": RandRed, "재밍": JamRed, "장악러시": RushRed, "혼합": MixRed, "rule": brains.RuleRed, "rl": brains.RLRed}
try:
    from agents.rl import QTable, RED_Q
    brains.use_rl(None, QTable.load(RED_Q))
except Exception:
    pass


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


def rollout(seed, red, default, switch_to, switch_at):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []
    for t in range(cfg["steps"]):
        cmp = run.compromised_drones(cyborg, n); ctx = {"compromised": cmp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(cmp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        kind = default if t < switch_at else switch_to
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(kind, int(a.split("_")[-1]), cmp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n


def cmean(red, default, switch_to, switch_at):
    return float(np.mean([rollout(s, red, default, switch_to, switch_at) for s in EVAL]))

# oracle-best loop per attack
best = {rn: min(LOOPS, key=lambda l: cmean(rc, l, l, 0)) for rn, rc in REDS.items()}
print("oracle-best loop:", best)

# sweep sensing delay D: pred for D steps then switch to oracle-best loop
print("\n=== compromise vs sensing delay D (switch to correct loop at step D) ===")
print("D".ljust(4) + "".join(f"{rn:>8}" for rn in REDS) + f"{'worst':>9}{'mean':>8}")
worst_by_D, mean_by_D, rows = [], [], []
jam_by_D = []
for D in DELAYS:
    vals = {rn: cmean(rc, "pred", best[rn], D) for rn, rc in REDS.items()}
    wc = max(vals.values()); mn = float(np.mean(list(vals.values())))
    worst_by_D.append(wc); mean_by_D.append(mn); jam_by_D.append(vals["재밍"])
    rows.append([D] + [round(vals[rn], 3) for rn in REDS] + [round(wc, 3), round(mn, 3)])
    print(f"{D:<4}" + "".join(f"{vals[rn]:8.2f}" for rn in REDS) + f"{wc:9.2f}{mn:8.2f}")

plt.figure(figsize=(8.5, 4.8))
plt.plot(DELAYS, worst_by_D, "o-", color="crimson", lw=2, label="worst-case (모든 공격)")
plt.plot(DELAYS, mean_by_D, "s-", color="navy", lw=1.8, label="평균")
plt.plot(DELAYS, jam_by_D, "^--", color="darkorange", lw=1.6, label="재밍(최난)")
plt.axhline(worst_by_D[0], ls=":", color="green", label=f"즉시감지=oracle({worst_by_D[0]:.2f})")
plt.xlabel("센싱 지연 D (공격을 늦게 식별할수록)"); plt.ylabel("최종 점령")
plt.title("센싱 속도가 적응 방어의 레버: 빨리 감지할수록 oracle에 가깝다")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig36_sensing.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_sensing.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["D"] + list(REDS) + ["worst", "mean"]); wr.writerows(rows)
print(f"\n즉시감지(D=0) worst={worst_by_D[0]:.3f} = oracle | 지연(D=10) worst={worst_by_D[DELAYS.index(10)]:.3f}")
print("Saved fig36_sensing.png, summary_sensing.csv")
