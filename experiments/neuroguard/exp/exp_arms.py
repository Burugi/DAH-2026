# -*- coding: utf-8 -*-
"""Two-sided adaptive arms race: the ATTACKER searches for the crack (best-response strategy
that maximizes compromise vs the current defense); the DEFENDER patches (best-response loop
that minimizes compromise vs that attack). Iterate. Does it converge or cycle?
Also reports whether a single adaptive defender (SMART) survives the crack-seeking attacker.
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
class NearRed(_R): SEQ = [2, 2, 6, 10]
class FarRed(_R):  SEQ = [4, 4, 4, 10]
class RandRed(_R): SEQ = [3, 3, 4, 10]
class JamRed(_R):  SEQ = [7, 8, 4, 6]
class RushRed(_R): SEQ = [2, 4, 6, 10, 10]
class MixRed(_R):  SEQ = [1, 4, 3, 6, 10, 7]
ATTACKS = {"근접": NearRed, "도약": FarRed, "랜덤": RandRed, "재밍": JamRed, "장악러시": RushRed, "혼합": MixRed}
DEFS = ["flat", "pred", "predOODA"]
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

class SmartMeta:
    W = 5
    def reset(self):
        self.cur = "pred"; self.phase = "exploit"; self.win0 = None; self.wt = 0; self.alts = ["flat", "predOODA"]; self.ai = 0; self.scores = {}
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


def comp(red_cls, defense):
    cs = []
    for s in EVAL:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, red_cls); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []
        meta = SmartMeta() if defense == "SMART" else None
        if meta: meta.reset()
        for t in range(cfg["steps"]):
            cmp = run.compromised_drones(cyborg, n); ctx = {"compromised": cmp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            hist.append(len(cmp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
            kind = meta.pick(len(cmp)) if meta else defense
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_act(kind, int(a.split("_")[-1]), cmp, pos, ml, rising), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if meta: meta.observe(len(run.compromised_drones(cyborg, n)))
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


# precompute attack x defense payoff matrix
A = list(ATTACKS); P = {(a, d): comp(ATTACKS[a], d) for a in A for d in DEFS}

# round-based best-response arms race (both adaptive)
defense = "pred"; rows = []; xs, ys, labs = [], [], []
print("=== adaptive arms race (attacker finds crack -> defender patches) ===")
for rnd in range(1, 7):
    atk = max(A, key=lambda a: P[(a, defense)]); c_atk = P[(atk, defense)]            # attacker best-response
    xs.append(rnd - 0.25); ys.append(c_atk); labs.append(f"R{rnd} 공격:{atk}")
    new_def = min(DEFS, key=lambda d: P[(atk, d)]); c_def = P[(atk, new_def)]          # defender best-response
    xs.append(rnd + 0.0); ys.append(c_def); labs.append(f"R{rnd} 방어:{new_def}")
    rows.append([rnd, atk, round(c_atk, 3), new_def, round(c_def, 3)])
    print(f"R{rnd}: 공격이 {defense} 틈 찾음 -> {atk}={c_atk:.3f}  |  방어 패치 -> {new_def}={c_def:.3f}")
    defense = new_def

# does the SMART single adaptive defender survive the crack-seeking attacker?
print("\n=== crack-seeking attacker vs each defense (worst attack per defense) ===")
worst = {}
for d in DEFS + ["SMART"]:
    cvals = {a: comp(ATTACKS[a], d) for a in A}
    wa = max(cvals, key=cvals.get); worst[d] = (wa, cvals[wa])
    print(f"  방어={d:9} 최악 공격={wa} -> {cvals[wa]:.3f}")

plt.figure(figsize=(11, 5))
plt.plot(xs, ys, "-o", color="purple", lw=1.8)
for x, y, l in zip(xs, ys, labs):
    plt.annotate(l, (x, y), fontsize=6, textcoords="offset points", xytext=(0, 7), ha="center")
plt.axhline(worst["SMART"][1], ls="--", color="crimson", label=f"SMART 적응방어 최악({worst['SMART'][1]:.2f})")
plt.xlabel("라운드 (공격 틈탐색 → 방어 패치)"); plt.ylabel("최종 점령")
plt.title("양측 적응 군비경쟁: 공격이 틈을 찾고 방어가 패치하는 순환")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig33_arms.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_arms.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["round", "attack", "comp_after_attack", "defense", "comp_after_patch"]); w.writerows(rows)
# detect cycle
defs_seq = [r[3] for r in rows]
print(f"\n방어 전환 시퀀스: {defs_seq}  -> {'순환(고정점 없음)' if len(set(defs_seq))>1 and defs_seq[0]!=defs_seq[-1] or len(set(defs_seq))>1 else '수렴'}")
print("Saved fig33_arms.png, summary_arms.csv")
