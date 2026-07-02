# -*- coding: utf-8 -*-
"""Mixed strategy: randomize pred vs block per drone/step (prob p). Game theory says a mixed
policy can lower EXPLOITABILITY (best-response attacker's gain) below any pure strategy.
Measure exploitability = worst compromise over an attack family, sweeping p."""
import sys, os, itertools
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]
WORM = {"near": 2, "far": 4, "rand": 3}
def make_red(vectors, tmode):
    class R(brains._Red):
        def get_action(self, obs, asp):
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5, obs)
            own = int(self.name.split("_")[-1]); v = vectors[own % len(vectors)]
            aid = int(self.np_random.choice([WORM[tmode], 6] if v == "W" else ([7, 8] if v == "J" else [9, 10])))
            return self._emit(aid, obs)
    return R
def frontier(i, comp, pos, ml): return any(d != i and np.linalg.norm(pos[i]-pos[d]) < ml for d in comp)
def rollout(seed, red, p):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 99)
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"]-1)]; live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        for a in live:
            i = int(a.split("_")[-1]); fr = frontier(i, comp, pos, ml)
            if i in comp: aid = 3
            elif comp and fr: aid = 6 if rng.random() < p else 8   # mix: block(p) vs decoy/pred(1-p)
            elif comp: aid = 4
            else: aid = 1
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n
def ev(red, p): return float(np.mean([rollout(s, red, p) for s in EVAL]))
FAMILY = [(list(c), tm) for k in (1,2,3) for c in itertools.combinations("WJB", k) for tm in ("near","far","rand")]
PS = [0.0, 0.25, 0.5, 0.75, 1.0]
print("=== mixed strategy exploitability (block prob p) ===\np".ljust(6)+"exploitability(worst)   avg")
res=[]
for p in PS:
    vals = [ev(make_red(v, tm), p) for (v, tm) in FAMILY]
    worst = max(vals); avg = float(np.mean(vals)); res.append((p, worst, avg))
    print(f"{p:<6.2f}{worst:18.3f}{avg:10.3f}")
best = min(res, key=lambda r: r[1])
print(f"\n최저 exploitability: p={best[0]} -> {best[1]:.3f} (pure pred p=0: {res[0][1]:.3f}, pure block p=1: {res[-1][1]:.3f})")
with open(os.path.join(OUT, "summary_mixed.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["block_prob_p", "exploitability_worst", "avg"])
    for r in res: wr.writerow([r[0], round(r[1],3), round(r[2],3)])
print("Saved summary_mixed.csv")
