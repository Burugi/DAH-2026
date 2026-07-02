# -*- coding: utf-8 -*-
"""Combine loop structure x policy: PRED(predictive) x OODA(adaptive) x LEARN(learned action).
8 combos. PRED=preempt frontier(next victims); OODA=escalate(block frontier) when spread rising;
LEARN=learned action choice (richer state + availability reward) vs hand-coded. Eval vs rule/rl red.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv, itertools
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import run
from agents import brains, actions
from agents.rl import QTable, BLUE_Q, RED_Q

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
PRO = [4, 6, 8]; MU = 0.3
try:
    brains.use_rl(QTable.load(BLUE_Q), QTable.load(RED_Q))
except Exception:
    pass


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)

def inb(i, comp, pos, ml):
    return sum(1 for d in comp if d != i and np.linalg.norm(pos[i] - pos[d]) < ml)

def rstate(i, comp, pos, n, ml):
    return (1 if i in comp else 0, min(3, inb(i, comp, pos, ml)), min(3, int(len(comp) / n * 4)))


def decide(i, comp, pos, n, ml, pred, aggressive, learned, q):
    """One drone's action under (pred, ooda-aggressive, learned)."""
    if i in comp:
        return 3, None
    fr = frontier(i, comp, pos, ml)
    if pred and fr:                                   # predictive: act on next victims
        if learned: idx = q_pick(q, i, comp, pos, n, ml); return PRO[idx], (rstate(i, comp, pos, n, ml), idx)
        return (6 if aggressive else 8), None         # block if escalating else decoy
    if comp:                                          # reactive on existing threat
        if aggressive and fr: return 6, None          # OODA escalate: block frontier
        if learned: idx = q_pick(q, i, comp, pos, n, ml); return PRO[idx], (rstate(i, comp, pos, n, ml), idx)
        return 4, None                                # retake
    return 1, None                                    # monitor


_RNG = [None]
def q_pick(q, i, comp, pos, n, ml):
    eps, rng = _RNG[0] if _RNG[0] else (0.0, None)
    return q.act(rstate(i, comp, pos, n, ml), eps, rng)


def run_combo(cfg, seed, red, pred, ooda, learned, q, train=False):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red)
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []; total = 0.0; trajs = {}
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        aggressive = ooda and rising
        live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}; blk = 0
        for a in live:
            i = int(a.split("_")[-1]); aid, pk = decide(i, comp, pos, n, ml, pred, aggressive, learned, q)
            if aid in (6, 8): blk += 1
            if pk is not None: picks[a] = pk
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts); r = (float(np.mean(list(rew.values()))) if rew else 0.0)
        total += r
        if train:
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, r - MU * blk))
        if all(done.values()): break
    if train:
        return trajs
    return len(run.compromised_drones(cyborg, n)) / n, total


def train_q(pred, ooda, episodes=220):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    for ep in range(episodes):
        _RNG[0] = (max(0.05, 1.0 - ep / (0.8 * episodes)), rng)
        trajs = run_combo(BASE, ep, brains.RuleRed, pred, ooda, True, q, train=True)
        for tr in trajs.values(): q.learn(tr)
    _RNG[0] = None
    return q


def evaluate(pred, ooda, learned, q, red):
    rs = [run_combo(BASE, s, red, pred, ooda, learned, q) for s in EVAL]
    return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))


REDS = [("rule", brains.RuleRed), ("rl", brains.RLRed)]
rows, comp_r, comp_l = [], {}, {}
print("=== combos: PRED x OODA x LEARN (compromise vs rule | rl) ===")
print(f"{'combo':10}{'rule':>9}{'rl':>9}{'reward(rule)':>14}")
for pred, ooda, learned in itertools.product([0, 1], [0, 1], [0, 1]):
    q = train_q(pred, ooda) if learned else None
    cr, rr = evaluate(pred, ooda, learned, q, brains.RuleRed)
    cl, rl = evaluate(pred, ooda, learned, q, brains.RLRed)
    name = ("P" if pred else "·") + ("O" if ooda else "·") + ("L" if learned else "·")
    comp_r[name] = cr; comp_l[name] = cl
    rows.append([name, pred, ooda, learned, round(cr, 3), round(cl, 3), round(rr, 1)])
    print(f"{name:10}{cr:9.3f}{cl:9.3f}{rr:14.0f}")

# sort by rule compromise for the plot
order = sorted(comp_r, key=lambda k: comp_r[k])
x = np.arange(len(order)); w = 0.38
plt.figure(figsize=(11, 5))
plt.bar(x - w/2, [comp_r[k] for k in order], w, label="vs rule-red", color="steelblue")
plt.bar(x + w/2, [comp_l[k] for k in order], w, label="vs rl-red", color="indianred")
plt.xticks(x, order, fontsize=10); plt.ylabel("최종 점령 (낮을수록 좋음)")
plt.title("루프×정책 8조합: 예측(P)·OODA(O)·학습(L)  [·=off]")
for i, k in enumerate(order):
    plt.text(i - w/2, comp_r[k] + 0.005, f"{comp_r[k]:.2f}", ha="center", fontsize=7)
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig28_combo.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_combo.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["combo", "PRED", "OODA", "LEARN", "comp_rule", "comp_rl", "reward_rule"]); wr.writerows(rows)
best = min(comp_r, key=lambda k: comp_r[k])
print(f"\nBEST vs rule: {best} = {comp_r[best]:.3f}")
print("Saved fig28_combo.png, summary_combo.csv")
