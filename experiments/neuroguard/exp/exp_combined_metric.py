# -*- coding: utf-8 -*-
"""Adopt the team's COMBINED metric (report.md, unweighted simple average) and compute it for our
full defense ladder, so our extended results speak the same language as the team dashboard.

  공격 점수 A = mean( 점령비율, 점령지속도(AUC), 1 - 첫침투시점/스텝, 1 - 웜탐지F1 )
  방어 점수 D = mean( 1-점령비율, 1-점령지속도, 웜탐지F1, 가용성 )

Note: the bundled summary.csv (old sweep id) does NOT reproduce this documented formula, so we
recompute consistently with the report definition. comp_F1 (worm detector) = 1.0 in this sim
(matches their summary where comp_F1=1.0 for every row). availability = mean_t over steps of the
fraction of drones that are neither compromised nor cut off by our own blocking (self-inflicted
availability cost -> rewards lean, surgical defense, per the competition's availability multiplier).
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv, itertools
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [3000, 3001, 3002, 3003, 3004]
STEPS = cfg["steps"]
COMP_F1 = 1.0                    # worm detector accuracy (=1.0 in this sim, matches team summary)
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            own = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[own % len(self.VECS)]])), obs)
    return ComboRed


ATTACKS = {"웜": make_combo_red(["W"]), "재밍": make_combo_red(["J"]), "차단": make_combo_red(["B"]),
           "동시(W+J+B)": make_combo_red(["W", "J", "B"]), "rule웜": brains.RuleRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(kind, i, comp, pos, ml, rising):
    if kind == "nodef": return 1
    if i in comp: return 3
    fr = frontier(i, comp, pos, ml)
    if kind == "flat": return 4 if comp else 1
    if kind == "pred": return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "predOODA":
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    if kind == "composite":
        if comp and fr: return 6
        if comp: return 4
        return 7
    if kind == "adaptive":
        if rising: return 6 if fr else (4 if comp else 1)
        return 4 if comp else 7
    return 1


def rollout(seed, red, defense):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    init = len(run.compromised_drones(cyborg, n))
    comp_frac, avail_frac, hist = [], [], []
    ttfc = None
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        cf = len(comp) / n
        comp_frac.append(cf)
        if ttfc is None and len(comp) > init:
            ttfc = t
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        blocks = 0; acts = {}
        for a in live:
            aid = blue_act(defense, int(a.split("_")[-1]), comp, pos, ml, rising)
            if aid == 6: blocks += 1                 # self-inflicted availability cost (own blocking)
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        avail_frac.append(max(0.0, (n - len(comp) - blocks) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = comp_frac[-1]
    auc = float(np.mean(comp_frac))
    ttfc_n = (ttfc if ttfc is not None else cfg["steps"]) / cfg["steps"]
    availability = float(np.mean(avail_frac))
    A = float(np.mean([final, auc, 1 - ttfc_n, 1 - COMP_F1]))
    D = float(np.mean([1 - final, 1 - auc, COMP_F1, availability]))
    return A, D, final, availability


def ev(red, defense):
    rs = [rollout(s, red, defense) for s in EVAL]
    return tuple(float(np.mean([r[k] for r in rs])) for k in range(4))   # A, D, final, avail


DEFENSES = ["nodef", "flat", "pred", "predOODA", "composite", "adaptive"]
A_grid = {d: {} for d in DEFENSES}; D_grid = {d: {} for d in DEFENSES}
print("=== combined metric (team formula) per defense x attack:  A(공격) | D(방어) ===")
for an, red in ATTACKS.items():
    for d in DEFENSES:
        A, D, fc, av = ev(red, d); A_grid[d][an] = A; D_grid[d][an] = D
    print(f"\n[{an}]")
    for d in DEFENSES:
        print(f"   {d:10}  A={A_grid[d][an]:.3f}  D={D_grid[d][an]:.3f}")

print("\n=== DEFENSE leaderboard (mean D over attack suite, higher=better) ===")
board = sorted(DEFENSES, key=lambda d: -np.mean([D_grid[d][a] for a in ATTACKS]))
for d in board:
    mD = float(np.mean([D_grid[d][a] for a in ATTACKS]))
    mA = float(np.mean([A_grid[d][a] for a in ATTACKS]))      # avg attacker score AGAINST this defense
    print(f"   {d:10}  방어점수 D={mD:.3f}   (상대 공격 A={mA:.3f})")

with open(os.path.join(OUT, "summary_combined_metric.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["defense"] + [f"{a}_A" for a in ATTACKS] + [f"{a}_D" for a in ATTACKS] + ["mean_D", "mean_A"])
    for d in DEFENSES:
        wr.writerow([d] + [round(A_grid[d][a], 3) for a in ATTACKS] + [round(D_grid[d][a], 3) for a in ATTACKS]
                    + [round(float(np.mean([D_grid[d][a] for a in ATTACKS])), 3),
                       round(float(np.mean([A_grid[d][a] for a in ATTACKS])), 3)])
print("\nSaved summary_combined_metric.csv")
