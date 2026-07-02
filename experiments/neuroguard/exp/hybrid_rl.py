# -*- coding: utf-8 -*-
"""HYBRID defender = heuristic CORE + learned ELSE.
Answers: "can an agentic system embed the heuristic and add learned consideration
so it is >= heuristic?"  Construction:
  - if own drone compromised      -> RemoveSessions (heuristic core, kept, critical)
  - else                          -> learned Q picks the action (decoy/block/retake/monitor)
Compare heuristic(multi) vs RL-multi vs HYBRID, all vs rule-red, held-out seeds.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run
from agents import brains, actions
from agents.rl import QTable, BLUE_Q

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL_SEEDS = [1000, 1001, 1002, 1003, 1004]
PROACTIVE = [4, 6, 8]   # learned ELSE chooses only among sensible defenses: Retake / Block / Decoy
                        # (Retake=4 is the rule's own move -> RL can always recover rule = floor)


def train_hybrid(cfg, episodes=300):
    """own-compromised -> RemoveSessions (heuristic core); else -> learned choice within PROACTIVE."""
    q = QTable(len(PROACTIVE))
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed)
        n = fleet["n"]; trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts, picks = {}, {}
            for a in live:
                own = int(a.split("_")[-1])
                if own in comp:
                    aid = 3                                   # heuristic core (no learning)
                else:
                    s = brains.blue_state(a, ctx)
                    idx = q.act(s, eps, rng)                  # learned else (within PROACTIVE)
                    aid = PROACTIVE[idx]
                    picks[a] = (s, idx)
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            r = float(np.mean(list(rew.values()))) if rew else 0.0
            for a, (s, aid) in picks.items():
                trajs.setdefault(a, []).append((s, aid, r))
            if all(done.values()):
                break
        for tr in trajs.values():
            q.learn(tr)
        if (ep + 1) % 75 == 0:
            print(f"  hybrid train {ep+1}/{episodes}  |states|={len(q.q)}")
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
        comps.append(len(run.compromised_drones(cyborg, n)) / n); rews.append(total); curves.append(curve)
    tmin = min(len(c) for c in curves)
    return np.mean(comps), np.std(comps), np.mean(rews), np.mean([c[:tmin] for c in curves], axis=0)


print("Training HYBRID (heuristic core + learned else) ...")
q_hybrid = train_hybrid(cfg)
q_multi = QTable.load(BLUE_Q)

def blue_heur(env, a, ctx):   return brains.blue_decide("rule", env, a, ctx)
def blue_multi(env, a, ctx):  return q_multi.act(brains.blue_state(a, ctx), 0.0, None)
def blue_hybrid(env, a, ctx):
    own = int(a.split("_")[-1])
    if own in ctx["compromised"]:
        return 3
    return PROACTIVE[q_hybrid.act(brains.blue_state(a, ctx), 0.0, None)]

defenders = {"heuristic\n(multi)": blue_heur, "RL-multi\n(per-drone Q)": blue_multi,
             "HYBRID\n(heuristic+RL)": blue_hybrid}

print(f"\n=== vs rule-red, held-out seeds {EVAL_SEEDS} ===")
print(f"{'defender':22} {'final_comp':>11} {'reward':>9}")
res = {}
for name, fn in defenders.items():
    c, cs, r, curve = eval_blue(cfg, fn, EVAL_SEEDS)
    res[name] = (c, cs, r, curve)
    print(f"{name.replace(chr(10),' '):22} {c:10.3f} {r:9.0f}")

labels = list(res.keys())
comp = [res[k][0] for k in labels]; cstd = [res[k][1] for k in labels]; rew = [res[k][2] for k in labels]
fig, ax1 = plt.subplots(figsize=(7.4, 4.6))
x = np.arange(len(labels))
ax1.bar(x, comp, yerr=cstd, color=["seagreen", "indianred", "royalblue"], alpha=0.85, capsize=4)
ax1.set_ylabel("final compromised fraction (lower=better)")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
for i, v in enumerate(comp): ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2 = ax1.twinx(); ax2.plot(x, rew, "o--", color="navy", lw=1.6)
ax2.set_ylabel("blue reward (higher=better)", color="navy")
plt.title("Hybrid (heuristic core + learned else) vs heuristic vs RL (vs rule-red, held-out)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig10_hybrid.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_hybrid.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["defender", "final_compromise", "std", "reward"])
    for k in labels: w.writerow([k.replace("\n", " "), round(res[k][0], 3), round(res[k][1], 3), round(res[k][2], 1)])
print("\nSaved fig10_hybrid.png, summary_hybrid.csv")
