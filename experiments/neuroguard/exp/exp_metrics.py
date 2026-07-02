# -*- coding: utf-8 -*-
"""Evaluation anchors + normalized metrics for the paper.
Baseline ladder  : no-defense -> random -> heuristic(flat) -> ours(pred/adaptive) -> oracle.
Attack suite     : worm / jam / block / simultaneous(W+J+B) / rule-worm(held-out-ish).
Metrics computed : mean+-std compromise & reward (5 seeds), gap-closed %, worst-case, exploitability.
"""
import sys, os, itertools
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [3000, 3001, 3002, 3003, 3004]
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            own = int(self.name.split("_")[-1])
            v = self.VECS[own % len(self.VECS)]
            return self._emit(int(self.np_random.choice(VEC_AIDS[v])), obs)
    return ComboRed


ATTACKS = {"웜": make_combo_red(["W"]), "재밍": make_combo_red(["J"]), "차단": make_combo_red(["B"]),
           "동시(W+J+B)": make_combo_red(["W", "J", "B"]), "rule웜(held-out)": brains.RuleRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(kind, i, comp, pos, ml, rising, rng):
    if kind == "nodef":
        return 1
    if kind == "random":
        return int(rng.integers(1, 9))                    # random decision 1..8
    if i in comp: return 3
    fr = frontier(i, comp, pos, ml)
    if kind == "flat":
        return 4 if comp else 1
    if kind == "pred":
        return 8 if (comp and fr) else (4 if comp else 1)
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
    rng = np.random.default_rng(seed + 7)
    hist = []; rsum = 0.0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(defense, int(a.split("_")[-1]), comp, pos, ml, rising, rng), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        rsum += float(np.mean(list(rew.values()))) if rew else 0.0
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, rsum


def ev(red, defense):
    cs, rs = [], []
    for s in EVAL:
        c, r = rollout(s, red, defense); cs.append(c); rs.append(r)
    return float(np.mean(cs)), float(np.std(cs)), float(np.mean(rs))


DEFENSES = ["nodef", "random", "flat", "pred", "predOODA", "composite", "adaptive"]
REAL_DEF = ["flat", "pred", "predOODA", "composite", "adaptive"]   # achievable defenses (for oracle)

C = {d: {} for d in DEFENSES}; S = {d: {} for d in DEFENSES}; R = {d: {} for d in DEFENSES}
print("=== compromise mean+-std | reward, per defense x attack ===")
for an, red in ATTACKS.items():
    for d in DEFENSES:
        c, sd, r = ev(red, d); C[d][an] = c; S[d][an] = sd; R[d][an] = r
    print(f"\n[{an}]")
    for d in DEFENSES:
        print(f"   {d:10} {C[d][an]:.3f} ± {S[d][an]:.3f}  | rew {R[d][an]:7.0f}")

# ---- oracle (best achievable real defense per attack) + normalized gap-closed ----
oracle = {an: min(C[d][an] for d in REAL_DEF) for an in ATTACKS}
nodef = {an: C["nodef"][an] for an in ATTACKS}
print("\n=== normalized metrics ===")
print("defense".ljust(11) + "gap-closed%   worst-case   avg-compromise")
metrics = {}
for d in REAL_DEF + ["random"]:
    gaps = []
    for an in ATTACKS:
        denom = nodef[an] - oracle[an]
        g = (nodef[an] - C[d][an]) / denom if denom > 1e-9 else 1.0
        gaps.append(g)
    gc = 100 * float(np.mean(gaps))
    worst = max(C[d][an] for an in ATTACKS)          # exploitability = best-response attacker's achievement
    avg = float(np.mean([C[d][an] for an in ATTACKS]))
    metrics[d] = (gc, worst, avg)
    print(f"{d:11}{gc:9.0f}    {worst:9.3f}    {avg:11.3f}")

print("\n=== anchors ===")
print(f"  무방어(floor) 평균 점령:  {np.mean([nodef[a] for a in ATTACKS]):.3f}")
print(f"  오라클(ceiling) 평균 점령: {np.mean([oracle[a] for a in ATTACKS]):.3f}")

with open(os.path.join(OUT, "summary_metrics.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["attack"] + [f"{d}_점령" for d in DEFENSES] + [f"{d}_std" for d in DEFENSES] + ["oracle", "nodef"])
    for an in ATTACKS:
        wr.writerow([an] + [round(C[d][an], 3) for d in DEFENSES] + [round(S[d][an], 3) for d in DEFENSES]
                    + [round(oracle[an], 3), round(nodef[an], 3)])
    wr.writerow([])
    wr.writerow(["defense", "gap_closed_%", "worst_case", "avg_compromise"])
    for d, (gc, w, a) in metrics.items():
        wr.writerow([d, round(gc, 1), round(w, 3), round(a, 3)])
print("\nSaved summary_metrics.csv")
