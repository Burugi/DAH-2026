# -*- coding: utf-8 -*-
"""CAGE-style network visualization of the attack/defense sim, step by step.
Runs a matchup, then draws the drone swarm as a NETWORK GRAPH per step:
  nodes = drones (triangle=UAV, square=UGV), edges = comms links (within range),
  blue = defended, red = compromised, purple ring = jammed, orange arrow = GPS spoofed.
Saves per-step frames and an 8-step montage for two contrasting matchups.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import run
from agents import brains, actions
from agents.rl import QTable, BLUE_Q

OUT = r"C:\workspace\DAH2026_exp\render"
os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
MAXLINK = cfg["fleet"].get("max_link", 40)
NUAV = cfg["fleet"]["n_uav"]
try:
    brains.use_rl(QTable.load(BLUE_Q), None)
except Exception:
    pass


def capture(seed, red_brain, blue_type):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red_brain)
    n = fleet["n"]; states = []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(brains.blue_decide(blue_type, env, a, ctx), env, a, ctx) for a in live}
        ti = min(t, fleet["steps"] - 1)
        states.append((fleet["pos_true"][ti].copy(), fleet["pos_rep"][ti].copy(),
                       set(comp), fleet["label_jam"][ti].copy(), fleet["label_gps"][ti].copy()))
        _, rew, done, _ = env.step(acts)
        if all(done.values()):
            break
    return fleet, states


def draw(ax, st, n, title):
    pos, rep, comp, jam, gps = st
    # edges = comms links within range (the network mesh)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < MAXLINK:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "-", color="0.85", lw=0.5, zorder=1)
    for i in range(n):
        c = "crimson" if i in comp else "steelblue"
        mk = "^" if i < NUAV else "s"
        ax.scatter(pos[i, 0], pos[i, 1], c=c, marker=mk, s=70, zorder=3, edgecolors="k", linewidths=0.4)
        if jam[i]:
            ax.scatter(pos[i, 0], pos[i, 1], facecolors="none", edgecolors="purple", marker="o", s=240, lw=1.6, zorder=4)
        if gps[i]:
            ax.annotate("", xy=(rep[i, 0], rep[i, 1]), xytext=(pos[i, 0], pos[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5), zorder=4)
    ax.set_title(title, fontsize=9); ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
    ax.set_xticks([]); ax.set_yticks([])


def montage(name, seed, red_brain, blue_type, n_steps=8):
    fleet, states = capture(seed, red_brain, blue_type)
    n = fleet["n"]; T = len(states)
    steps = sorted(set(int(round(x)) for x in np.linspace(0, T - 1, n_steps)))
    # individual frames
    fdir = os.path.join(OUT, name); os.makedirs(fdir, exist_ok=True)
    for t in range(T):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        nc = len(states[t][2])
        draw(ax, states[t], n, f"{name}  step {t}   compromised {nc}/{n}")
        fig.tight_layout(); fig.savefig(os.path.join(fdir, f"frame_{t:02d}.png"), dpi=110); plt.close(fig)
    # montage
    cols = 4; rows = int(np.ceil(len(steps) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for k, t in enumerate(steps):
        ax = axs[k // cols][k % cols]
        nc = len(states[t][2])
        draw(ax, states[t], n, f"step {t}   compromised {nc}/{n}")
    for k in range(len(steps), rows * cols):
        axs[k // cols][k % cols].axis("off")
    legend = [Line2D([0], [0], marker="^", color="w", markerfacecolor="steelblue", markersize=10, label="UAV (defended)"),
              Line2D([0], [0], marker="s", color="w", markerfacecolor="steelblue", markersize=10, label="UGV (defended)"),
              Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson", markersize=10, label="compromised"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="purple", markersize=12, label="jammed"),
              Line2D([0], [0], color="darkorange", lw=2, label="GPS spoof"),
              Line2D([0], [0], color="0.8", lw=1, label="comms link")]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8, frameon=False)
    title = {"rule": "heuristic", "rl": "RL", "llm": "LLM"}.get(blue_type, blue_type)
    fig.suptitle(f"공방 시뮬레이션 — red={red_brain.__name__.replace('Red','')} vs blue={title}  "
                 f"(최종 감염 {len(states[-1][2])}/{n})", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(OUT, f"montage_{name}.png"), dpi=120); plt.close(fig)
    print(f"{name}: {T} steps, final compromised {len(states[-1][2])}/{n} -> montage_{name}.png ({len(steps)} frames shown, {T} saved)")


# Two contrasting matchups: defense FAILS (worm spreads) vs defense WORKS (contained)
montage("rule_vs_RL_defense_fails", 1000, brains.RuleRed, "rl")
montage("rule_vs_heuristic_contained", 1000, brains.RuleRed, "rule")
print("Saved to", OUT)
