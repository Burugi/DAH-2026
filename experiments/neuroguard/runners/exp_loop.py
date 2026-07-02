# -*- coding: utf-8 -*-
"""Agentic LOOP STRUCTURE comparison (not the policy/state/reward, but the loop itself).
Same action repertoire (heuristic floor + retake/block/decoy/monitor), different LOOP:
  L0 flat-reactive  : memoryless, per-step independent (current).
  L1 OODA-adaptive  : Orient(spread trend) -> Decide(posture: aggressive if rising / conservative
                      if contained, availability-aware) -> Act. Closed feedback loop.
  L2 predictive     : defend the FRONTIER (clean drones adjacent to infected = next victims) preemptively.
  L3 full-agentic   : OODA posture + predictive + memory combined.
Action ids: 1 Monitor, 3 RemoveSessions, 4 Retake, 6 Block, 8 Decoy.
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
from agents.rl import QTable, BLUE_Q, RED_Q

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
try:
    brains.use_rl(QTable.load(BLUE_Q), QTable.load(RED_Q))
except Exception:
    pass


def inf_nbrs(i, comp, pos, ml):
    return [d for d in comp if d != i and np.linalg.norm(pos[i] - pos[d]) < ml]


class L0_Flat:
    name = "L0 평면 반응형"
    def reset(self): pass
    def decide(self, comp, pos, n, ml, ids, t):
        out = {}
        for i in ids:
            out[i] = 3 if i in comp else (4 if comp else 1)
        return out


class L1_OODA:
    name = "L1 OODA-적응(폐루프)"
    def reset(self): self.hist = []
    def decide(self, comp, pos, n, ml, ids, t):
        self.hist.append(len(comp))
        rising = len(self.hist) >= 3 and self.hist[-1] > self.hist[-3]      # Orient: spread trend
        aggressive = rising                                                  # Decide: posture
        out = {}
        for i in ids:
            if i in comp: out[i] = 3                                         # Act: clean own
            elif aggressive and inf_nbrs(i, comp, pos, ml): out[i] = 6       # block frontier when spreading
            elif comp: out[i] = 4                                            # retake
            else: out[i] = 1
        return out


class L2_Predict:
    name = "L2 예측형(선제)"
    def reset(self): pass
    def decide(self, comp, pos, n, ml, ids, t):
        out = {}
        for i in ids:
            if i in comp: out[i] = 3
            elif inf_nbrs(i, comp, pos, ml): out[i] = 8                       # preempt next victims (decoy)
            elif comp: out[i] = 4
            else: out[i] = 1
        return out


class L3_Full:
    name = "L3 풀 에이전틱(OODA+예측+메모리)"
    def reset(self): self.hist = []; self.dur = {}
    def decide(self, comp, pos, n, ml, ids, t):
        self.hist.append(len(comp))
        for i in list(self.dur):                                            # memory: infection duration
            if i not in comp: self.dur.pop(i, None)
        for i in comp: self.dur[i] = self.dur.get(i, 0) + 1
        rising = len(self.hist) >= 3 and self.hist[-1] > self.hist[-3]
        aggressive = rising or (len(comp) / n > 0.4)
        out = {}
        for i in ids:
            if i in comp: out[i] = 3
            elif inf_nbrs(i, comp, pos, ml):                                 # frontier
                out[i] = 6 if aggressive else 8                              # block if spreading else decoy
            elif comp: out[i] = 4
            else: out[i] = 1
        return out


def rollout(cfg, seed, loop, red):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    loop.reset(); total = 0.0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        ids = {int(a.split("_")[-1]): a for a in live}
        dec = loop.decide(comp, pos, n, ml, list(ids.keys()), t)
        acts = {ids[i]: actions.make_blue_index(dec[i], env, ids[i], ctx) for i in ids}
        _, rew, done, _ = env.step(acts); total += float(np.mean(list(rew.values()))) if rew else 0.0
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, total


def evaluate(cfg, loop_cls, seeds, red):
    rs = [rollout(cfg, s, loop_cls(), red) for s in seeds]
    return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))


LOOPS = [L0_Flat, L1_OODA, L2_Predict, L3_Full]
REDS = [("rule-red", brains.RuleRed), ("rl-red", brains.RLRed)]
rows, comp_t, rew_t = [], {}, {}
print("=== loop structure comparison (compromise | reward, held-out) ===")
print("loop".ljust(34) + "".join(f"{rn:>20}" for rn, _ in REDS))
for L in LOOPS:
    comp_t[L.name] = []; rew_t[L.name] = []; line = L.name.ljust(34)
    for rn, rb in REDS:
        c, r = evaluate(BASE, L, EVAL, rb); comp_t[L.name].append(c); rew_t[L.name].append(r); rows.append([L.name, rn, round(c, 3), round(r, 1)])
        line += f"  {c:.3f}|{r:>6.0f}".rjust(20)
    print(line)

labels = [L.name for L in LOOPS]; x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(11, 5))
plt.bar(x - w/2, [comp_t[l][0] for l in labels], w, label="vs rule-red", color="steelblue")
plt.bar(x + w/2, [comp_t[l][1] for l in labels], w, label="vs rl-red", color="indianred")
plt.xticks(x, labels, fontsize=8); plt.ylabel("최종 점령 (낮을수록 좋음)")
plt.title("에이전틱 루프 구조 비교 (같은 행동·다른 루프)")
for i, l in enumerate(labels):
    plt.text(i - w/2, comp_t[l][0] + 0.01, f"{comp_t[l][0]:.2f}", ha="center", fontsize=7)
    plt.text(i + w/2, comp_t[l][1] + 0.01, f"{comp_t[l][1]:.2f}", ha="center", fontsize=7)
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig27_loop.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_loop.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["loop", "red", "compromise", "reward"]); wr.writerows(rows)
print("\nReward (vs rule-red):", {l: round(rew_t[l][0]) for l in labels})
print("Saved fig27_loop.png, summary_loop.csv")
