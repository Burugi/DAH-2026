# -*- coding: utf-8 -*-
"""Co-evolution arms race: attack creativity vs defense ceiling.
Each round: (1) the attacker (rl-red) LEARNS to beat the current defense -> compromise rises
(attack breaks the ceiling); (2) the defender picks its best-response loop vs the adapted
attacker -> compromise drops (defense recovers). Repeat -> oscillating arms-race curve.
"""
import sys, os
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
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [2000, 2001, 2002, 2003]
ROUNDS = 5; RED_EP = 90
BLUES = ["flat", "pred", "predOODA"]


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)

def blue_act(kind, i, comp, pos, ml, rising):
    if i in comp: return 3
    if kind == "flat":
        return 4 if comp else 1
    fr = frontier(i, comp, pos, ml)
    if kind == "pred":
        if comp and fr: return 8
        return 4 if comp else 1
    if kind == "predOODA":
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    return 1


def play(seed, blue_kind, q_red, train_red=False, q_for_train=None):
    if q_red is not None: brains.set_red_q(q_red, q_for_train[0] if train_red else 0.0,
                                           q_for_train[1] if train_red else None, train=train_red)
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RLRed)
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); hist = []; trajs = {}
    prev = len(run.compromised_drones(cyborg, n))
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(blue_kind, int(a.split("_")[-1]), comp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if train_red:
            stash = brains.pop_red_stash(); now = len(run.compromised_drones(cyborg, n)); r = float(now - prev); prev = now
            for name, s, aid in stash: trajs.setdefault(name, []).append((s, aid, r))
        if all(done.values()): break
    if train_red: brains.set_red_q(None, 0.0, None, False)
    return len(run.compromised_drones(cyborg, n)) / n, trajs


def train_red_vs(blue_kind, q_red, episodes=RED_EP, rnd=0):
    rng = np.random.default_rng(100 + rnd)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        _, trajs = play(1000 * rnd + ep, blue_kind, q_red, train_red=True, q_for_train=(eps, rng))
        for tr in trajs.values(): q_red.learn(tr)


def eval_blue(blue_kind, q_red):
    brains.set_red_q(q_red, 0.0, None, False)
    return float(np.mean([play(s, blue_kind, q_red)[0] for s in EVAL]))


# start: our optimal predictive defense, fresh attacker
blue = "pred"; q_red = QTable(actions.RED_N)
phase_x, phase_y, phase_lbl, picks = [], [], [], []
print("=== co-evolution arms race ===")
# initial: predictive defense vs naive attacker
c0 = eval_blue(blue, q_red); phase_x.append(0); phase_y.append(c0); phase_lbl.append("초기(예측방어)"); picks.append(blue)
print(f"round 0  init: blue={blue} comp={c0:.3f}")
for rnd in range(1, ROUNDS + 1):
    # 1) attacker adapts (learns to beat current defense) -> ceiling broken
    train_red_vs(blue, q_red, rnd=rnd)
    c_atk = eval_blue(blue, q_red)
    phase_x.append(rnd - 0.25); phase_y.append(c_atk); phase_lbl.append(f"R{rnd} 공격적응"); picks.append(blue)
    # 2) defender best-response among loops -> recovers
    scores = {k: eval_blue(k, q_red) for k in BLUES}
    blue = min(scores, key=scores.get); c_def = scores[blue]
    phase_x.append(rnd + 0.0); phase_y.append(c_def); phase_lbl.append(f"R{rnd} 방어적응({blue})"); picks.append(blue)
    print(f"round {rnd}: 공격적응 comp={c_atk:.3f} -> 방어 best={blue} comp={c_def:.3f}  scores={ {k:round(v,2) for k,v in scores.items()} }")

plt.figure(figsize=(10, 5))
plt.plot(phase_x, phase_y, "-o", color="purple", lw=1.8)
for x, y, l in zip(phase_x, phase_y, phase_lbl):
    plt.annotate(l, (x, y), fontsize=6, textcoords="offset points", xytext=(0, 6), ha="center")
# shade attack-up vs defense-down
plt.axhline(0.067, ls=":", color="green", label="방어 천장(고정공격 0.067)")
plt.xlabel("라운드 (공격 적응 → 방어 적응)"); plt.ylabel("최종 점령 비율")
plt.title("공격 창의성 vs 방어 천장: 공진화 군비경쟁")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig30_coevo.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_coevo.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["phase_x", "compromise", "label", "blue"])
    for x, y, l, b in zip(phase_x, phase_y, phase_lbl, picks): w.writerow([x, round(y, 3), l, b])
print("Saved fig30_coevo.png, summary_coevo.csv")
