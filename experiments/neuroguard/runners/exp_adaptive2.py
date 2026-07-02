# -*- coding: utf-8 -*-
"""Adaptive defense v2 — fix the jam weakness with a fast cascade.
SMART(v1) blind-explored all alternatives -> exploration cost. SMART2 switches to the NEXT loop
ONLY when the current one is failing, and STICKS while it contains (fast, W=3). Order
pred -> flat -> predOODA (flat is the robust fallback for jam/leap, predOODA for adaptive).
Compare SMART2 to v1 / oracle / fixed across all attacks; report worst-case (lower=more robust).
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


class SmartV1:
    W = 5
    def reset(self): self.cur = "pred"; self.phase = "exploit"; self.win0 = None; self.wt = 0; self.alts = ["flat", "predOODA"]; self.ai = 0; self.scores = {}
    def pick(self, c):
        if self.win0 is None: self.win0 = c
        return self.cur
    def observe(self, c):
        self.wt += 1
        if self.wt >= self.W:
            g = c - self.win0
            if self.phase == "exploit":
                if g > 0 and self.cur == "pred": self.phase = "explore"; self.ai = 0; self.cur = self.alts[0]; self.scores = {"pred": g}
            else:
                self.scores[self.cur] = g; self.ai += 1
                if self.ai < len(self.alts): self.cur = self.alts[self.ai]
                else: self.cur = min(self.scores, key=self.scores.get); self.phase = "exploit"
            self.wt = 0; self.win0 = c


class SmartV2:
    """Fast cascade: switch to next loop only while failing; stick while containing. W=3."""
    W = 3
    ORDER = ["pred", "flat", "predOODA"]
    def reset(self): self.idx = 0; self.cur = self.ORDER[0]; self.win0 = None; self.wt = 0
    def pick(self, c):
        if self.win0 is None: self.win0 = c
        return self.cur
    def observe(self, c):
        self.wt += 1
        if self.wt >= self.W:
            if c - self.win0 > 0:                          # current failing -> next loop
                self.idx = (self.idx + 1) % len(self.ORDER); self.cur = self.ORDER[self.idx]
            self.wt = 0; self.win0 = c


def rollout(seed, red, mode):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []
    meta = {"v1": SmartV1, "v2": SmartV2}.get(mode); meta = meta() if meta else None
    if meta: meta.reset()
    for t in range(cfg["steps"]):
        cmp = run.compromised_drones(cyborg, n); ctx = {"compromised": cmp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(cmp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        kind = meta.pick(len(cmp)) if meta else mode
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(kind, int(a.split("_")[-1]), cmp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if meta: meta.observe(len(run.compromised_drones(cyborg, n)))
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n

def ev(red, mode): return float(np.mean([rollout(s, red, mode) for s in EVAL]))


cols = ["oracle", "SMART v1", "SMART v2(캐스케이드)"]
print("attack".ljust(12) + "".join(f"{c:>20}" for c in cols))
rows = []; M = {}
for rn, rc in REDS.items():
    fixed = [ev(rc, l) for l in LOOPS]; oracle = min(fixed)
    v1 = ev(rc, "v1"); v2 = ev(rc, "v2")
    M[rn] = [oracle, v1, v2]; rows.append([rn, round(oracle, 3), round(v1, 3), round(v2, 3)])
    print(rn.ljust(12) + "".join(f"{v:20.3f}" for v in M[rn]))

print("\n=== worst-case across ALL attacks (낮을수록 견고) ===")
for j, c in enumerate(cols):
    wc = max(M[r][j] for r in REDS); print(f"  {c:22} worst={wc:.3f}")

reds = list(REDS); x = np.arange(len(reds)); w = 0.26
plt.figure(figsize=(12, 5))
for j, c in enumerate(cols):
    plt.bar(x + (j - 1) * w, [M[r][j] for r in reds], w, label=c, color=["#666666", "#d62728", "crimson"][j])
plt.xticks(x, reds, fontsize=8, rotation=15); plt.ylabel("최종 점령 (낮을수록 방어 성공)")
plt.title("적응방어 고도화 v2: 빠른 캐스케이드로 약점 보강")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig34_adaptive_v2.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_adaptive_v2.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack"] + cols); wr.writerows(rows)
print("Saved fig34_adaptive_v2.png, summary_adaptive_v2.csv")
