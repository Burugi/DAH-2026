# -*- coding: utf-8 -*-
"""Search for MAXIMUM defense performance across all levers.
 A) predictive action x defense-radius grid (decoy/block/retake x r1.0/r1.5/all).
 B) OODA escalation + aggressive cleanup variants on the best.
 C) learned predictive with FIXED reward (don't penalize PREVENTIVE block/decoy on frontier).
Report the global best vs rule-red and rl-red.
"""
import sys, os, itertools
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
PRO = [4, 6, 8]
try:
    brains.use_rl(QTable.load(BLUE_Q), QTable.load(RED_Q))
except Exception:
    pass


def near_infected(i, comp, pos, radius):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < radius for d in comp)

def inb(i, comp, pos, ml):
    return sum(1 for d in comp if d != i and np.linalg.norm(pos[i] - pos[d]) < ml)

def rstate(i, comp, pos, n, ml):
    return (1 if i in comp else 0, min(3, inb(i, comp, pos, ml)), min(3, int(len(comp) / n * 4)))


def run_loop(cfg, seed, red, decide):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red)
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []; total = 0.0; q_traj = {}
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]; acts = {}; picks = {}; waste = 0
        for a in live:
            i = int(a.split("_")[-1]); aid, pk = decide(i, comp, pos, n, ml, rising)
            if aid in (6, 8) and not near_infected(i, comp, pos, ml): waste += 1
            if pk is not None: picks[a] = pk
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts); total += float(np.mean(list(rew.values()))) if rew else 0.0
        for a, pk in picks.items(): q_traj.setdefault(a, []).append((pk[0], pk[1], rew, waste))
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, total, q_traj


def ev(decide, red):
    rs = [run_loop(BASE, s, red, decide) for s in EVAL]; return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))


def make_predictive(radius_mult, paction, ooda=False, aggressive_clean=False):
    def decide(i, comp, pos, n, ml, rising):
        if i in comp: return 3, None
        r = radius_mult * ml if radius_mult < 50 else 1e9
        if comp and near_infected(i, comp, pos, r):
            act = paction
            if ooda and rising and paction == 8: act = 6      # escalate decoy->block when spreading
            return act, None
        if comp: return 4, None
        return 1, None
    return decide


# ---------- A) action x radius grid ----------
print("=== A) predictive action x radius (compromise vs rule | rl) ===")
rows = []; results = {}
for rad, pa in itertools.product([1.0, 1.5, 99], [8, 6, 4]):
    d = make_predictive(rad, pa)
    cr, _ = ev(d, brains.RuleRed); cl, _ = ev(d, brains.RLRed)
    nm = {1.0: "r1.0", 1.5: "r1.5", 99: "all"}[rad] + "+" + {8: "decoy", 6: "block", 4: "retake"}[pa]
    results[nm] = (cr, cl); rows.append([nm, round(cr, 3), round(cl, 3)])
    print(f"  {nm:14} {cr:.3f} | {cl:.3f}")

# ---------- B) OODA + aggressive variants on best radius+decoy ----------
print("\n=== B) variants ===")
for nm, d in [("r1.5+decoy+OODA", make_predictive(1.5, 8, ooda=True)),
              ("all+decoy", make_predictive(99, 8)),
              ("all+block", make_predictive(99, 6))]:
    cr, _ = ev(d, brains.RuleRed); cl, _ = ev(d, brains.RLRed); results[nm] = (cr, cl); rows.append([nm, round(cr, 3), round(cl, 3)])
    print(f"  {nm:18} {cr:.3f} | {cl:.3f}")

# ---------- C) learned predictive with FIXED reward (no penalty on preventive actions) ----------
print("\n=== C) learned predictive (fixed reward: only waste penalized) ===")
def train_fixed(radius_mult=1.5, episodes=220, mu=0.3):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    def dec(i, comp, pos, n, ml, rising):
        if i in comp: return 3, None
        r = radius_mult * ml
        if comp and near_infected(i, comp, pos, r):
            s = rstate(i, comp, pos, n, ml); idx = q.act(s, dec.eps, rng); return PRO[idx], (s, idx)
        if comp: return 4, None
        return 1, None
    for ep in range(episodes):
        dec.eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        _, _, qt = run_loop(BASE, ep, brains.RuleRed, dec)
        for a, steps in qt.items():
            tr = [(s, idx, (float(np.mean(list(rw.values()))) if rw_ok(rw) else 0.0) - mu * waste) for (s, idx, rw, waste) in steps for rw_ok in [lambda x: bool(x)]]
            q.learn(tr)
    return q
def rw_mean(rw): return float(np.mean(list(rw.values()))) if rw else 0.0
def train_fixed2(radius_mult=1.5, episodes=220, mu=0.3):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    def dec(i, comp, pos, n, ml, rising):
        if i in comp: return 3, None
        if comp and near_infected(i, comp, pos, radius_mult * ml):
            s = rstate(i, comp, pos, n, ml); idx = q.act(s, dec.eps, rng); return PRO[idx], (s, idx)
        if comp: return 4, None
        return 1, None
    dec.eps = 0.1
    for ep in range(episodes):
        dec.eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        _, _, qt = run_loop(BASE, ep, brains.RuleRed, dec)
        for a, steps in qt.items():
            q.learn([(s, idx, rw_mean(rw) - mu * waste) for (s, idx, rw, waste) in steps])
    return q
qf = train_fixed2()
def dec_learned(i, comp, pos, n, ml, rising):
    if i in comp: return 3, None
    if comp and near_infected(i, comp, pos, 1.5 * ml):
        return PRO[qf.act(rstate(i, comp, pos, n, ml), 0.0, None)], None
    if comp: return 4, None
    return 1, None
cr, _ = ev(dec_learned, brains.RuleRed); cl, _ = ev(dec_learned, brains.RLRed)
results["learned-pred(fixed보상)"] = (cr, cl); rows.append(["learned-pred(fixed)", round(cr, 3), round(cl, 3)])
print(f"  learned-pred(fixed) {cr:.3f} | {cl:.3f}")

# ---------- report ----------
best_rule = min(results, key=lambda k: results[k][0]); best_rl = min(results, key=lambda k: results[k][1])
print(f"\n*** GLOBAL BEST vs rule: {best_rule} = {results[best_rule][0]:.3f}")
print(f"*** GLOBAL BEST vs rl  : {best_rl} = {results[best_rl][1]:.3f}")

order = sorted(results, key=lambda k: results[k][0])[:10]
x = np.arange(len(order)); w = 0.38
plt.figure(figsize=(12, 5))
plt.bar(x - w/2, [results[k][0] for k in order], w, label="vs rule-red", color="steelblue")
plt.bar(x + w/2, [results[k][1] for k in order], w, label="vs rl-red", color="indianred")
plt.xticks(x, order, fontsize=7, rotation=20); plt.ylabel("최종 점령 (낮을수록 좋음)")
plt.title("최고성능 탐색: 예측 행동×범위 + 변형 + 학습 (상위 10)")
for i, k in enumerate(order): plt.text(i - w/2, results[k][0] + 0.004, f"{results[k][0]:.3f}", ha="center", fontsize=6)
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig29_maxperf.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_maxperf.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["config", "comp_rule", "comp_rl"]); wr.writerows(rows)
print("Saved fig29_maxperf.png, summary_maxperf.csv")
