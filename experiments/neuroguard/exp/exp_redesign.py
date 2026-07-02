# -*- coding: utf-8 -*-
"""Redesign experiment: fix the flaws and test an improved 'optimal' structure.
 FIX1 richer LOCAL state: (own_compromised, #infected_neighbours, global_frac) -> agent sees topology.
 FIX2 availability-aware reward: penalize over-use of costly block/decoy during training.
 KEEP heuristic floor + per-drone (the two validated axes).
Compare {heuristic, RL-multi(old 8-state), RL-multi+(richer), hybrid(old), HYBRID+(richer+avail)}
against rule-red AND rl-red (held-out). Plus size generalization (12/18/30) for hybrid vs HYBRID+.
"""
import sys, os, copy, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import run
from agents import brains, actions
from agents.rl import QTable, BLUE_Q, RED_Q

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
PRO = [4, 6, 8]                          # retake / block / decoy
MU = 0.3                                  # availability penalty per costly (block/decoy) action
q_old = QTable.load(BLUE_Q)
try:
    brains.use_rl(q_old, QTable.load(RED_Q))
except Exception as e:
    print("warn rl:", e)


def inf_nbrs(i, comp, pos, maxlink):
    return sum(1 for d in comp if d != i and np.linalg.norm(pos[i] - pos[d]) < maxlink)

def rstate(i, comp, pos, n, maxlink):
    return (1 if i in comp else 0, min(3, inf_nbrs(i, comp, pos, maxlink)), min(3, int(len(comp) / n * 4)))


def rollout(cfg, seed, red_brain, decide):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red_brain)
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); total = 0.0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(decide(int(a.split("_")[-1]), comp, pos, n, ml, env, a, ctx), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts); total += float(np.mean(list(rew.values()))) if rew else 0.0
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, total


def evaluate(cfg, decide, seeds, red_brain):
    rs = [rollout(cfg, s, red_brain, decide) for s in seeds]
    return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))


def train(cfg, mode, episodes=260):
    """mode: 'multi+' (richer, no floor, full catalog) or 'hybrid+' (richer, floor, PRO, avail reward)."""
    nact = len(PRO) if mode == "hybrid+" else actions.BLUE_DECISION_N
    q = QTable(nact); rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed)
        n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}; blk = 0
            for a in live:
                i = int(a.split("_")[-1]); s = rstate(i, comp, pos, n, ml)
                if mode == "hybrid+" and i in comp:
                    aid = 3
                else:
                    idx = q.act(s, eps, rng); aid = PRO[idx] if mode == "hybrid+" else idx; picks[a] = (s, idx)
                if aid in (6, 8): blk += 1
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            r = (float(np.mean(list(rew.values()))) if rew else 0.0) - MU * blk      # availability-aware
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, r))
            if all(done.values()): break
        for tr in trajs.values(): q.learn(tr)
        if (ep + 1) % 65 == 0: print(f"  train {mode} {ep+1}/{episodes} |states|={len(q.q)}")
    return q


# also retrain the OLD 8-state hybrid for reference
def train_hybrid_old(cfg, episodes=260):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed); n = fleet["n"]; trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}
            for a in live:
                own = int(a.split("_")[-1])
                if own in comp: aid = 3
                else:
                    s = brains.blue_state(a, ctx); idx = q.act(s, eps, rng); aid = PRO[idx]; picks[a] = (s, idx)
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, float(np.mean(list(rew.values()))) if rew else 0.0))
            if all(done.values()): break
        for tr in trajs.values(): q.learn(tr)
    return q


print("Training models ...")
q_multi_plus = train(BASE, "multi+")
q_hyb_plus = train(BASE, "hybrid+")
q_hyb_old = train_hybrid_old(BASE)

def d_heur(i, comp, pos, n, ml, env, a, ctx): return brains.blue_decide("rule", env, a, ctx)
def d_rlmulti(i, comp, pos, n, ml, env, a, ctx): return q_old.act(brains.blue_state(a, ctx), 0.0, None)
def d_multiplus(i, comp, pos, n, ml, env, a, ctx): return q_multi_plus.act(rstate(i, comp, pos, n, ml), 0.0, None)
def d_hybold(i, comp, pos, n, ml, env, a, ctx):
    return 3 if i in comp else PRO[q_hyb_old.act(brains.blue_state(a, ctx), 0.0, None)]
def d_hybplus(i, comp, pos, n, ml, env, a, ctx):
    return 3 if i in comp else PRO[q_hyb_plus.act(rstate(i, comp, pos, n, ml), 0.0, None)]

MODELS = [("휴리스틱", d_heur), ("RL-멀티(구8상태)", d_rlmulti), ("RL-멀티+(풍부)", d_multiplus),
          ("하이브리드(구)", d_hybold), ("HYBRID+ (재설계)", d_hybplus)]
REDS = [("rule-red", brains.RuleRed), ("rl-red", brains.RLRed)]

print("\n=== compromise | reward  vs each attacker (held-out) ===")
rows = []; comp_tab = {}
hdr = "model".ljust(20) + "".join(f"{rn:>22}" for rn, _ in REDS)
print(hdr)
for mname, dfn in MODELS:
    line = mname.ljust(20); comp_tab[mname] = []
    for rn, rb in REDS:
        c, r = evaluate(BASE, dfn, EVAL, rb); comp_tab[mname].append(c); rows.append([mname, rn, round(c, 3), round(r, 1)])
        line += f"  {c:.3f}|{r:>7.0f}".rjust(22)
    print(line)

# fig: grouped bars (compromise) per attacker
labels = [m for m, _ in MODELS]; x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(10, 5))
plt.bar(x - w/2, [comp_tab[m][0] for m in labels], w, label="vs rule-red", color="steelblue")
plt.bar(x + w/2, [comp_tab[m][1] for m in labels], w, label="vs rl-red", color="indianred")
plt.xticks(x, labels, fontsize=8); plt.ylabel("최종 점령 (낮을수록 좋음)")
plt.title("재설계 비교: 풍부 지역상태 + 가용성 보상 (다중 공격자, held-out)")
for i, m in enumerate(labels):
    plt.text(i - w/2, comp_tab[m][0] + 0.01, f"{comp_tab[m][0]:.2f}", ha="center", fontsize=7)
    plt.text(i + w/2, comp_tab[m][1] + 0.01, f"{comp_tab[m][1]:.2f}", ha="center", fontsize=7)
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig23_redesign.png"), dpi=130); plt.close()

# size generalization: old hybrid vs HYBRID+ (does richer state reduce overfit?)
print("\n=== size generalization (vs rule-red): hybrid(old) vs HYBRID+ ===")
sizes = [(8, 4), (12, 6), (20, 10)]; sg = {"하이브리드(구)": [], "HYBRID+ (재설계)": []}
print("size      hybrid(old)   HYBRID+")
for nu, ng in sizes:
    cfg = copy.deepcopy(BASE); cfg["fleet"]["n_uav"] = nu; cfg["fleet"]["n_ugv"] = ng; cfg["attacks"] = []
    co, _ = evaluate(cfg, d_hybold, EVAL, brains.RuleRed); cp, _ = evaluate(cfg, d_hybplus, EVAL, brains.RuleRed)
    sg["하이브리드(구)"].append(co); sg["HYBRID+ (재설계)"].append(cp)
    print(f"{nu+ng:>4}        {co:.3f}        {cp:.3f}")
plt.figure(figsize=(7, 4.3))
for k, col in [("하이브리드(구)", "royalblue"), ("HYBRID+ (재설계)", "purple")]:
    plt.plot([nu+ng for nu, ng in sizes], sg[k], "o-", color=col, lw=1.9, label=k)
plt.xlabel("군집 크기"); plt.ylabel("최종 점령"); plt.title("크기 일반화: 풍부 지역상태가 과적합을 줄이나")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig24_redesign_size.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_redesign.csv"), "w", newline="", encoding="utf-8") as f:
    w_ = csv.writer(f); w_.writerow(["model", "red", "compromise", "reward"]); w_.writerows(rows)
print("\nSaved fig23_redesign.png, fig24_redesign_size.png, summary_redesign.csv")
