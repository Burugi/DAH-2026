# -*- coding: utf-8 -*-
"""Adaptive META-defense: an online loop that FINDS a defense for an UNKNOWN attack.
It doesn't know the attack. It watches the EFFECT (is compromise contained under the current
loop?) and, if the current loop is failing, it switches/searches among candidate loops
(explore) until it finds one that contains the spread (exploit); re-explores if it starts
failing again. Tested vs every attack strategy and compared to fixed loops + the oracle
(best fixed loop per attack).
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
REDS = {"근접확산": NearRed, "도약(원거리)": FarRed, "랜덤+원거리": RandRed,
        "재밍+확산": JamRed, "장악러시": RushRed, "혼합창의": MixRed,
        "rule": brains.RuleRed, "rl(학습)": brains.RLRed}
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


class AdaptiveMeta:
    """Online bandit over defense loops, driven by observed containment (compromise growth)."""
    W = 4
    def reset(self):
        self.growth = {}; self.cur = 0; self.phase = "explore"; self.expi = 0; self.wt = 0; self.win0 = None; self.best = None
    def pick(self, comp_count):
        if self.win0 is None: self.win0 = comp_count
        return LOOPS[self.cur] if self.phase == "explore" else self.best
    def observe(self, comp_count):
        self.wt += 1
        if self.wt >= self.W:
            g = comp_count - self.win0
            if self.phase == "explore":
                self.growth[LOOPS[self.expi]] = g; self.expi += 1
                if self.expi >= len(LOOPS):
                    self.best = min(self.growth, key=self.growth.get); self.phase = "exploit"
                else:
                    self.cur = self.expi
            else:
                if g > 0:                                   # best loop now failing -> re-search
                    self.phase = "explore"; self.expi = 0; self.cur = 0; self.growth = {}
            self.wt = 0; self.win0 = comp_count


class SmartMeta:
    """Effect-based: default to the best general loop (pred); only if it is FAILING (compromise
    rising over a window) explore alternatives, then keep the best. No blind upfront exploration."""
    W = 5
    def reset(self):
        self.cur = "pred"; self.phase = "exploit"; self.win0 = None; self.wt = 0
        self.alts = ["predOODA", "flat"]; self.ai = 0; self.scores = {}
    def pick(self, comp_count):
        if self.win0 is None: self.win0 = comp_count
        return self.cur
    def observe(self, comp_count):
        self.wt += 1
        if self.wt >= self.W:
            g = comp_count - self.win0
            if self.phase == "exploit":
                if g > 0 and self.cur == "pred":            # pred failing -> reactively explore
                    self.phase = "explore"; self.ai = 0; self.cur = self.alts[0]; self.scores = {"pred": g}
            else:
                self.scores[self.cur] = g; self.ai += 1
                if self.ai < len(self.alts): self.cur = self.alts[self.ai]
                else: self.cur = min(self.scores, key=self.scores.get); self.phase = "exploit"
            self.wt = 0; self.win0 = comp_count


def rollout(seed, red, mode):
    """mode in LOOPS (fixed) | 'adaptive'(naive bandit) | 'smart'(effect-based)."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []
    meta = None
    if mode == "adaptive": meta = AdaptiveMeta(); meta.reset()
    elif mode == "smart": meta = SmartMeta(); meta.reset()
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        kind = meta.pick(len(comp)) if meta else mode
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(kind, int(a.split("_")[-1]), comp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if meta: meta.observe(len(run.compromised_drones(cyborg, n)))
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n


def ev(red, mode):
    return float(np.mean([rollout(s, red, mode) for s in EVAL]))


print("=== adaptive meta-defense vs fixed loops (final compromise) ===")
cols = LOOPS + ["oracle(최선고정)", "순진adaptive", "SMART(효과기반)"]
print("attack".ljust(14) + "".join(f"{c:>15}" for c in cols))
rows = []; M = {}
for rn, rc in REDS.items():
    fixed = {l: ev(rc, l) for l in LOOPS}
    oracle = min(fixed.values())
    adap = ev(rc, "adaptive"); smart = ev(rc, "smart")
    M[rn] = [fixed[l] for l in LOOPS] + [oracle, adap, smart]
    rows.append([rn] + [round(v, 3) for v in M[rn]])
    print(rn.ljust(14) + "".join(f"{v:16.3f}" for v in M[rn]))

# worst-case across attacks for each strategy
print("\n=== worst-case across ALL attacks (낮을수록 견고) ===")
for j, c in enumerate(cols):
    wc = max(M[r][j] for r in REDS); print(f"  {c:18} worst={wc:.3f}")

reds = list(REDS.keys()); x = np.arange(len(reds)); w = 0.13
COLcols = ["#9ecae1", "#fdae6b", "#a1d99b", "#666666", "#d62728", "crimson"]
plt.figure(figsize=(13, 5.2))
for j, c in enumerate(cols):
    plt.bar(x + (j - 2.5) * w, [M[r][j] for r in reds], w, label=c, color=COLcols[j])
plt.xticks(x, reds, fontsize=8, rotation=15); plt.ylabel("최종 점령 (낮을수록 방어 성공)")
plt.title("적응형 메타 방어: 모르는 공격에도 온라인으로 최선 방어를 찾는다")
plt.legend(fontsize=8, ncol=5); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig32_adaptive.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_adaptive.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack"] + cols); wr.writerows(rows)
print("Saved fig32_adaptive.png, summary_adaptive.csv")
