# -*- coding: utf-8 -*-
"""Controlled blue-policy sweep on the worm sim (E1 sim-side, CybORG required).
Same red config (RedDroneWorm, start=3, spawn=0.20); vary blue.policy over all 5;
measure worm spread (compromised) + reward over 5 seeds. Also a spread-over-time curve.
"""
import sys, os, copy
sys.path.insert(0, r"C:\workspace\DAH-2026")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run import rollout

OUT = r"C:\workspace\DAH2026_exp"
base = yaml.safe_load(open(r"C:\workspace\DAH-2026\configs\scenario_defended.yaml", encoding="utf-8"))
policies = ["sleep", "random", "remove", "retake", "react"]
seeds = [0, 1, 2, 3, 4]

mean_comp, std_comp, mean_rew, curves = {}, {}, {}, {}
print("=== Blue-policy sweep (same RedDroneWorm: start=3 spawn=0.20, 5 seeds) ===")
print(f"{'policy':8} {'compromised(/18)':>18} {'totReward':>12}")
for pol in policies:
    cfg = copy.deepcopy(base); cfg["blue"]["policy"] = pol
    comp, rew, spread = [], [], []
    for s in seeds:
        fleet, reward, red_owned, dfn = rollout(cfg, s)
        comp.append(int(red_owned[-1].sum())); rew.append(float(reward.sum()))
        spread.append(red_owned.sum(axis=1))           # compromised count per step
    Tmin = min(len(x) for x in spread)
    curves[pol] = np.mean([x[:Tmin] for x in spread], axis=0)
    mean_comp[pol] = np.mean(comp); std_comp[pol] = np.std(comp); mean_rew[pol] = np.mean(rew)
    print(f"{pol:8} {np.mean(comp):8.1f} +/- {np.std(comp):3.1f}      {np.mean(rew):10.0f}")

# Fig6: compromised + reward by policy
fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
x = np.arange(len(policies))
ax1.bar(x, [mean_comp[p] for p in policies], yerr=[std_comp[p] for p in policies],
        color="indianred", alpha=0.85, capsize=4)
ax1.set_ylabel("compromised drones (/18)", color="indianred")
ax1.set_xticks(x); ax1.set_xticklabels(policies)
for i, p in enumerate(policies): ax1.text(i, mean_comp[p]+0.3, f"{mean_comp[p]:.1f}", ha="center", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(x, [mean_rew[p] for p in policies], "o-", color="navy", lw=1.8)
ax2.set_ylabel("total reward (higher=better)", color="navy")
plt.title("E1 sim-side: blue policy vs worm spread & reward\n(same RedDroneWorm, 5 seeds)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_blue_policy.png"), dpi=130); plt.close()

# Fig7: worm spread over time, sleep vs react
plt.figure(figsize=(7, 4.2))
for pol, col in [("sleep", "crimson"), ("retake", "darkorange"), ("react", "seagreen")]:
    plt.plot(curves[pol], color=col, lw=1.9, label=f"blue={pol} (final {mean_comp[pol]:.1f}/18)")
plt.xlabel("sim step"); plt.ylabel("compromised drones")
plt.title("Worm propagation over time: active defense flattens the spread")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig7_worm_spread.png"), dpi=130); plt.close()

import csv
with open(os.path.join(OUT, "summary_sim_policy.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["policy", "compromised_mean", "compromised_std", "reward_mean"])
    for p in policies: w.writerow([p, round(mean_comp[p], 2), round(std_comp[p], 2), round(mean_rew[p], 1)])
print("\nSaved fig6_blue_policy.png, fig7_worm_spread.png, summary_sim_policy.csv to", OUT)
