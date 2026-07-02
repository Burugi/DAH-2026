# -*- coding: utf-8 -*-
"""Sweep every remaining axis x defender type (rule/llm/rl), rule-red attacker.
Axes: starting infections, comms range (topology), episode length, grid density.
Worm channel only (attacks=[]). Final compromise, mean over seeds.
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
cfg0 = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
cfg0["attacks"] = []                       # worm channel only
try:
    brains.use_rl(QTable.load(BLUE_Q), QTable.load(RED_Q))
except Exception as e:
    print("warn rl:", e)
BLUES = ["rule", "llm", "rl"]
SEEDS = [1000, 1001, 1002, 1003]


def final_comp(cfg, blue_type, seeds):
    cs = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, brains.RuleRed); n = fleet["n"]
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(brains.blue_decide(blue_type, env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


# (axis label, setter(cfg,val), values)
def set_start(c, v): c["sim"]["starting_num_red"] = v
def set_links(c, v): c["sim"]["max_length_data_links"] = v; c["fleet"]["max_link"] = v
def set_steps(c, v): c["steps"] = v
def set_grid(c, v): c["fleet"]["grid"] = v

AXES = [
    ("시작 감염 드론 수", set_start, [1, 2, 3, 5, 8]),
    ("통신 사거리(토폴로지 밀도)", set_links, [20, 30, 40, 60, 80]),
    ("에피소드 길이(steps)", set_steps, [20, 40, 60, 80, 120]),
    ("격자 크기(드론 밀도↓)", set_grid, [60, 100, 140, 180]),
]

rows = []
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for ax, (name, setter, vals) in zip(axs.ravel(), AXES):
    print(f"\n=== axis: {name} ===")
    print("  val   " + "  ".join(f"{b:>6}" for b in BLUES))
    series = {b: [] for b in BLUES}
    for v in vals:
        cfg = copy.deepcopy(cfg0); setter(cfg, v)
        line = []
        for b in BLUES:
            fc = final_comp(cfg, b, SEEDS); series[b].append(fc); line.append(fc)
            rows.append([name, v, b, round(fc, 3)])
        print(f"  {v:>5} " + "  ".join(f"{x:6.2f}" for x in line))
    for b, col in zip(BLUES, ["seagreen", "darkorange", "indianred"]):
        ax.plot(vals, series[b], "o-", color=col, lw=1.9, label=f"blue={b}")
    ax.set_xlabel(name); ax.set_ylabel("최종 점령 비율"); ax.set_title(name); ax.set_ylim(-0.03, 1.0); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle("남은 축 전체 sweep (rule-red, 방어 3종, worm 채널)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(OUT, "fig22_axes_sweep.png"), dpi=120); plt.close(fig)

with open(os.path.join(OUT, "summary_axes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["axis", "value", "blue", "final_compromise"]); w.writerows(rows)
print(f"\nSaved fig22_axes_sweep.png, summary_axes.csv  ({len(rows)} data points)")
