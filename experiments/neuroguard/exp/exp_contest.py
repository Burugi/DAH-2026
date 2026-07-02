# -*- coding: utf-8 -*-
"""Two-sided contest: red WORM spread vs blue RECLAIM spread, running simultaneously.
 (A) per-step flow: how many nodes red newly seizes vs blue reclaims each step (tug-of-war).
 (B) equilibrium heatmap: attacker strength (spawn rate) x defender strength (reaction prob)
     -> where the front line stabilizes (final compromise).
 (C) contested montage: nodes flipping red<->blue highlighted, the moving front.
"""
import sys, os, random, copy
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.lines import Line2D
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp\render"; os.makedirs(OUT, exist_ok=True)
cfg0 = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
MAXLINK = cfg0["fleet"].get("max_link", 40); NUAV = cfg0["fleet"]["n_uav"]


def pdef(p, seed=0):
    """Parametric defender: act with the rule policy with prob p each call, else Sleep (strength knob)."""
    rng = random.Random(seed)
    def fn(env, a, ctx):
        return brains.blue_decide("rule", env, a, ctx) if rng.random() < p else 0
    return fn


def run_contest(cfg, seed, blue_fn):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RuleRed); n = fleet["n"]
    prev = run.compromised_drones(cyborg, n)
    red_gain, blue_gain, net, hist, flips = [], [], [], [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        rg = comp - prev; bg = prev - comp                       # red seizes / blue reclaims
        red_gain.append(len(rg)); blue_gain.append(len(bg)); net.append(len(comp))
        hist.append(set(comp)); flips.append((set(rg), set(bg)))
        prev = set(comp)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_fn(env, a, ctx), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return dict(red_gain=red_gain, blue_gain=blue_gain, net=net, hist=hist, flips=flips, fleet=fleet, n=n)


def final_comp(cfg, blue_fn, seeds):
    cs = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, brains.RuleRed); n = fleet["n"]
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_fn(env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


# ===== (A) per-step tug-of-war flow (balanced fight) =====
cfg = copy.deepcopy(cfg0); cfg["sim"]["red_spawn_rate"] = 0.20
R = run_contest(cfg, 1000, pdef(0.6))
T = len(R["net"]); x = np.arange(T)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(x, R["red_gain"], color="crimson", label="red seizes (worm spread)")
ax.bar(x, [-b for b in R["blue_gain"]], color="steelblue", label="blue reclaims (defense spread)")
ax.plot(x, R["net"], "k-o", ms=3, lw=1.5, label="net compromised")
ax.axhline(0, color="0.5", lw=0.7)
ax.set_xlabel("step"); ax.set_ylabel("nodes  (+red seizes / -blue reclaims)")
ax.set_title("(A) 양측 동시 확산 줄다리기: red 점령 vs blue 재장악 (spawn=0.20, def=0.6)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig17_tugofwar.png"), dpi=130); plt.close(fig)
print(f"(A) tug-of-war: total red seizes={sum(R['red_gain'])}, blue reclaims={sum(R['blue_gain'])}, final net={R['net'][-1]}/{R['n']}")

# ===== (B) equilibrium heatmap: attacker x defender strength -> final compromise =====
spawns = [0.05, 0.10, 0.20, 0.35]
defs = [0.0, 0.25, 0.5, 0.75, 1.0]
M = np.zeros((len(spawns), len(defs)))
for i, sp in enumerate(spawns):
    c = copy.deepcopy(cfg0); c["sim"]["red_spawn_rate"] = sp
    for j, pd in enumerate(defs):
        M[i, j] = final_comp(c, pdef(pd), [1000, 1001, 1002])
fig, ax = plt.subplots(figsize=(7, 4.6))
im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(defs))); ax.set_xticklabels([f"{d:.2f}" for d in defs])
ax.set_yticks(range(len(spawns))); ax.set_yticklabels([f"{s:.2f}" for s in spawns])
ax.set_xlabel("defender strength (reaction prob)"); ax.set_ylabel("attacker strength (worm spawn rate)")
for i in range(len(spawns)):
    for j in range(len(defs)):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if M[i, j] > 0.5 else "black")
fig.colorbar(im, ax=ax, label="final compromised fraction")
ax.set_title("(B) 공방 균형: 공격강도 x 방어강도 -> 전선(최종 점령) 위치")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig18_equilibrium.png"), dpi=130); plt.close(fig)
print("(B) equilibrium heatmap saved. diagonal balance visible.")

# ===== (C) contested montage: flips highlighted (the moving front) =====
def draw(ax, st, flips, fleet, n, title):
    pos = fleet["pos_true"][min(st_idx, fleet["steps"] - 1)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < MAXLINK:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "-", color="0.88", lw=0.5, zorder=1)
    newred, newblue = flips
    for i in range(n):
        c = "crimson" if i in st else "steelblue"
        mk = "^" if i < NUAV else "s"
        ax.scatter(pos[i, 0], pos[i, 1], c=c, marker=mk, s=70, zorder=3, edgecolors="k", linewidths=0.4)
        if i in newred:
            ax.scatter(pos[i, 0], pos[i, 1], facecolors="none", edgecolors="red", marker="o", s=300, lw=2.2, zorder=5)
        if i in newblue:
            ax.scatter(pos[i, 0], pos[i, 1], facecolors="none", edgecolors="deepskyblue", marker="o", s=300, lw=2.2, zorder=5)
    ax.set_title(title, fontsize=9); ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_xticks([]); ax.set_yticks([])

steps = sorted(set(int(round(v)) for v in np.linspace(0, T - 1, 8)))
fig, axs = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
for k, st_idx in enumerate(steps):
    ax = axs[k // 4][k % 4]
    draw(ax, R["hist"][st_idx], R["flips"][st_idx], R["fleet"], R["n"],
         f"step {st_idx}  net {R['net'][st_idx]}/{R['n']}  (+{R['red_gain'][st_idx]}red /-{R['blue_gain'][st_idx]}blue)")
leg = [Line2D([0], [0], marker="o", color="w", markeredgecolor="red", markerfacecolor="none", ms=12, label="newly seized (red)"),
       Line2D([0], [0], marker="o", color="w", markeredgecolor="deepskyblue", markerfacecolor="none", ms=12, label="newly reclaimed (blue)"),
       Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson", ms=10, label="compromised"),
       Line2D([0], [0], marker="^", color="w", markerfacecolor="steelblue", ms=10, label="defended")]
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9, frameon=False)
fig.suptitle("(C) 경합 몽타주: 노드가 빨강↔파랑으로 뒤집히는 전선 (spawn=0.20, def=0.6)", fontsize=13)
fig.tight_layout(rect=[0, 0.04, 1, 0.97]); fig.savefig(os.path.join(OUT, "fig19_contest_montage.png"), dpi=115); plt.close(fig)
print("(C) contest montage saved.")
print("Saved fig17/18/19 to", OUT)
