# -*- coding: utf-8 -*-
"""Add the missing SINGLE-AGENT (centralized) RL and compare:
   heuristic(multi)  vs  RL-multi(per-drone shared Q, code team)  vs  RL-single(central global Q).
All blue defenders vs the same rule-red attacker, held-out seeds. Built on the code-team harness.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run
from agents import brains, actions
from agents.rl import QTable, BLUE_Q

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
N_TRAIN = 200
EVAL_SEEDS = [1000, 1001, 1002, 1003, 1004]   # held-out (training used 0..199)


def global_state(comp, n):
    """Centralized single-agent state: only the global compromise-fraction bucket."""
    frac = len(comp) / max(1, n)
    return (min(3, int(frac * 4)),)


def train_central(cfg, episodes=N_TRAIN):
    q = QTable(actions.BLUE_DECISION_N)
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed)
        n = fleet["n"]; traj = []
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            s = global_state(comp, n)
            aid = q.act(s, eps, rng)                     # ONE decision for the whole swarm
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(aid, env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            r = float(np.mean(list(rew.values()))) if rew else 0.0
            traj.append((s, aid, r))
            if all(done.values()):
                break
        q.learn(traj)
        if (ep + 1) % 50 == 0:
            print(f"  central train {ep+1}/{episodes}  |states|={len(q.q)}")
    return q


def eval_blue(cfg, blue_fn, seeds):
    comps, rews, curves = [], [], []
    for seed in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RuleRed)
        n = fleet["n"]; total = 0.0; curve = []
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_fn(env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            total += float(np.mean(list(rew.values()))) if rew else 0.0
            curve.append(len(run.compromised_drones(cyborg, n)))
            if all(done.values()):
                break
        comps.append(len(run.compromised_drones(cyborg, n)) / n); rews.append(total)
        curves.append(curve)
    tmin = min(len(c) for c in curves)
    return np.mean(comps), np.std(comps), np.mean(rews), np.mean([c[:tmin] for c in curves], axis=0)


print("Training RL-single (centralized global Q) vs rule-red ...")
q_central = train_central(cfg)
q_multi = QTable.load(BLUE_Q)        # code team's per-drone shared Q (already trained)

blues = {
    "heuristic\n(multi)":  lambda env, a, ctx: brains.blue_decide("rule", env, a, ctx),
    "RL-multi\n(per-drone Q)": lambda env, a, ctx: q_multi.act(brains.blue_state(a, ctx), 0.0, None),
    "RL-single\n(central Q)":  lambda env, a, ctx: q_central.act(global_state(ctx["compromised"], ctx["n"]), 0.0, None),
}

print(f"\n=== vs rule-red, held-out seeds {EVAL_SEEDS} ===")
print(f"{'blue defender':22} {'final_comp':>11} {'reward':>9}")
res = {}
for name, fn in blues.items():
    c, cs, r, curve = eval_blue(cfg, fn, EVAL_SEEDS)
    res[name] = (c, cs, r, curve)
    print(f"{name.replace(chr(10),' '):22} {c:10.3f} {r:9.0f}")

# Fig8: single vs multi vs heuristic
labels = list(res.keys())
comp = [res[k][0] for k in labels]; cstd = [res[k][1] for k in labels]; rew = [res[k][2] for k in labels]
fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
x = np.arange(len(labels))
b = ax1.bar(x, comp, yerr=cstd, color=["seagreen", "indianred", "darkorange"], alpha=0.85, capsize=4)
ax1.set_ylabel("final compromised fraction (lower=better)")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
for i, v in enumerate(comp): ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2 = ax1.twinx(); ax2.plot(x, rew, "o--", color="navy", lw=1.6)
ax2.set_ylabel("blue reward (higher=better)", color="navy")
plt.title("Single-agent vs Multi-agent vs Heuristic (defender, vs rule-red, held-out)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig8_single_vs_multi.png"), dpi=130); plt.close()

# Fig9: compromise curves
plt.figure(figsize=(7.2, 4.3))
for name, col in zip(labels, ["seagreen", "indianred", "darkorange"]):
    plt.plot(res[name][3], lw=1.9, color=col, label=f"{name.replace(chr(10),' ')} (final {res[name][0]:.2f})")
plt.xlabel("step"); plt.ylabel("# compromised drones")
plt.title("Worm containment over time: heuristic vs RL-multi vs RL-single")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig9_single_multi_curve.png"), dpi=130); plt.close()

import csv
with open(os.path.join(OUT, "summary_single_vs_multi.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["defender", "final_compromise", "std", "reward"])
    for k in labels: w.writerow([k.replace("\n", " "), round(res[k][0], 3), round(res[k][1], 3), round(res[k][2], 1)])
print("\nSaved fig8_single_vs_multi.png, fig9_single_multi_curve.png, summary_single_vs_multi.csv")
