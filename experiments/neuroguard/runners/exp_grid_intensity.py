# -*- coding: utf-8 -*-
"""Full grid: every attacker x defender matchup across attack intensities.
 red {rule, llm, rl} x blue {rule, llm, rl} x spawn_rate {0.05, 0.10, 0.20, 0.35}
 = 36 matchups. Final compromise (mean over seeds). One 3x3 heatmap per intensity
 + a summary line (defender compromise vs intensity, averaged over attackers).
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
import agents.rl as rlmod
from agents.rl import QTable, BLUE_Q, RED_Q

OUT = r"C:\workspace\DAH2026_exp"
cfg0 = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
try:
    brains.use_rl(QTable.load(BLUE_Q), QTable.load(RED_Q))
except Exception as e:
    print("warn: rl Q load:", e)

REDS = ["rule", "llm", "rl"]
BLUES = ["rule", "llm", "rl"]
SPAWNS = [0.05, 0.10, 0.20, 0.35]
SEEDS = [1000, 1001, 1002, 1003]


def final_comp(cfg, red_type, blue_type, seeds):
    red_brain = brains.RED_BRAINS[red_type]
    cs = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, red_brain); n = fleet["n"]
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(brains.blue_decide(blue_type, env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


grids, rows = {}, []
print("=== full grid: red x blue x spawn (final compromise) ===")
for sp in SPAWNS:
    cfg = copy.deepcopy(cfg0); cfg["sim"]["red_spawn_rate"] = sp
    M = np.zeros((len(REDS), len(BLUES)))
    for i, r in enumerate(REDS):
        for j, b in enumerate(BLUES):
            M[i, j] = final_comp(cfg, r, b, SEEDS)
            rows.append([sp, r, b, round(M[i, j], 3)])
    grids[sp] = M
    print(f"spawn={sp}:\n  red\\blue " + "  ".join(f"{b:>5}" for b in BLUES))
    for i, r in enumerate(REDS):
        print(f"  {r:7} " + "  ".join(f"{M[i,j]:5.2f}" for j in range(len(BLUES))))

# 4 heatmaps (one per intensity)
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for ax, sp in zip(axs.ravel(), SPAWNS):
    M = grids[sp]; im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(BLUES))); ax.set_xticklabels(BLUES); ax.set_yticks(range(len(REDS))); ax.set_yticklabels(REDS)
    ax.set_xlabel("blue (방어)"); ax.set_ylabel("red (공격)"); ax.set_title(f"spawn rate = {sp}")
    for i in range(len(REDS)):
        for j in range(len(BLUES)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if M[i, j] > 0.55 else "black")
fig.colorbar(im, ax=axs.ravel().tolist(), label="최종 점령 비율", fraction=0.04)
fig.suptitle("강도×매치업 전조합: red×blue 3×3 × spawn 4단계 (최종 점령)", fontsize=14)
fig.savefig(os.path.join(OUT, "fig20_grid_intensity.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# summary: defender compromise vs intensity (avg over attackers)
plt.figure(figsize=(7, 4.4))
for b, col in zip(BLUES, ["seagreen", "darkorange", "indianred"]):
    ys = [np.mean([grids[sp][i, BLUES.index(b)] for i in range(len(REDS))]) for sp in SPAWNS]
    plt.plot(SPAWNS, ys, "o-", color=col, lw=1.9, label=f"blue={b}")
plt.xlabel("공격 강도 (worm spawn rate)"); plt.ylabel("최종 점령 (공격자 평균)")
plt.title("강도별 방어자 성능 (공격 3종 평균)")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig21_intensity_summary.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_grid_intensity.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["spawn", "red", "blue", "final_compromise"]); w.writerows(rows)
print("\nSaved fig20_grid_intensity.png, fig21_intensity_summary.png, summary_grid_intensity.csv")
