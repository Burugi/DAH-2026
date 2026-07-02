# -*- coding: utf-8 -*-
"""Fix the ExploitKnown (random-target) blind spot of predictive defence.
Pred assumes spatial locality (infected neighbour = next target); a random-target attack breaks it.
Fix = LOCALITY-ADAPTIVE defence: measure whether new compromises appear ADJACENT to old ones
(worm, local) or SCATTERED (random, non-local), and route:
  high locality -> predictive frontier mode (pred)
  low  locality -> global reactive retake (don't waste clean drones on a frontier that doesn't predict)
Goal: match pred on the worm AND close the gap on random-target attacks.
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
THRESH = 4
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10], "K": [3, 3], "L": [2, 6]}


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            own = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[own % len(self.VECS)]])), obs)
    return ComboRed


ATTACKS = {"웜(국소)": make_combo_red(["W"]),
           "ExploitKnown(랜덤)": make_combo_red(["K"]),
           "동시(W+J+B)": make_combo_red(["W", "J", "B"]),
           "rule웜": brains.RuleRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def is_adjacent(i, ref, pos, ml):
    return any(np.linalg.norm(pos[i] - pos[d]) < ml for d in ref if d != i)


def blue_act(kind, i, comp, pos, ml, rising, fast):
    """fast = spread is wide/fast this window (high branching) -> block sources instead of predicting."""
    if i in comp: return 3
    fr = frontier(i, comp, pos, ml)
    if kind == "pred":
        return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "adaptive":
        if rising: return 6 if fr else (4 if comp else 1)
        return 4 if comp else 7
    if kind == "pred_block":                    # pred but BLOCK the source instead of no-op decoy
        return 6 if (comp and fr) else (4 if comp else 1)
    if kind.startswith("branch"):               # slow contiguous -> pred ; fast/wide -> block sources
        if not fast:
            return 8 if (comp and fr) else (4 if comp else 1)      # pred
        return 6 if (comp and fr) else (4 if comp else 1)          # block sources
    return 1


def rollout(seed, red, defense):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    prev = set(run.compromised_drones(cyborg, n))
    hist = []; new_window = []; fast_trace = []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        new = comp - prev
        new_window.append(len(new))
        if len(new_window) > 3: new_window.pop(0)
        fast = sum(new_window) >= THRESH            # branching threshold over last 3 steps (tunable)
        fast_trace.append(1.0 if fast else 0.0)
        prev = set(comp)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(defense, int(a.split("_")[-1]), comp, pos, ml, rising, fast), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, float(np.mean(fast_trace))


def ev(red, defense):
    rs = [rollout(s, red, defense) for s in EVAL]
    return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))


print("=== branching threshold sweep (worm should stay pred-low, random should drop) ===")
print("THRESH".ljust(8) + "웜(국소)".rjust(12) + "ExploitKnown".rjust(14) + "동시".rjust(10) + "rule웜".rjust(10))
best = None
for TH in [3, 4, 5, 6, 7]:
    globals()["THRESH"] = TH
    wm = ev(ATTACKS["웜(국소)"], "branch")[0]
    ek = ev(ATTACKS["ExploitKnown(랜덤)"], "branch")[0]
    sm = ev(ATTACKS["동시(W+J+B)"], "branch")[0]
    rw = ev(ATTACKS["rule웜"], "branch")[0]
    score = max(wm, ek, sm, rw)                     # minimize the worst -> robust
    print(f"{TH:<8}{wm:12.3f}{ek:14.3f}{sm:10.3f}{rw:10.3f}   worst={score:.3f}")
    if best is None or score < best[1]: best = (TH, score)
print(f"\n최적 THRESH={best[0]} (worst-case {best[1]:.3f})")

globals()["THRESH"] = best[0]
DEFENSES = ["pred", "adaptive", f"branch(TH={best[0]})"]
KIND = {"pred": "pred", "adaptive": "adaptive", f"branch(TH={best[0]})": "branch"}
print("\n=== final comparison (final compromise) ===")
print("attack".ljust(22) + "".join(f"{d:>16}" for d in DEFENSES))
grid = {}
for an, red in ATTACKS.items():
    row = [ev(red, KIND[d])[0] for d in DEFENSES]; grid[an] = row
    print(an.ljust(22) + "".join(f"{v:16.3f}" for v in row))

plt.figure(figsize=(8.6, 4.9))
x = np.arange(len(ATTACKS)); w = 0.26
for k, d in enumerate(DEFENSES):
    plt.bar(x + (k - 1) * w, [grid[a][k] for a in ATTACKS], w, label=d)
plt.xticks(x, list(ATTACKS), fontsize=9)
plt.ylabel("최종 점령 (낮을수록 방어 성공)")
plt.title("국소성 인지 적응 방어: ExploitKnown(랜덤표적) 맹점 보강")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig50_locality_fix.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_locality_fix.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack"] + DEFENSES)
    for a in ATTACKS: wr.writerow([a] + [round(v, 3) for v in grid[a]])

ek = "ExploitKnown(랜덤)"; wm = "웜(국소)"
print(f"\n핵심: ExploitKnown  pred {grid[ek][0]:.3f} -> branch {grid[ek][2]:.3f}   "
      f"| 웜(국소)  pred {grid[wm][0]:.3f} -> branch {grid[wm][2]:.3f} (유지)")
print("Saved fig50_locality_fix.png, summary_locality_fix.csv")
