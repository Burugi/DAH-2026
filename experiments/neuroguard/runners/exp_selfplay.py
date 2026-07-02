# -*- coding: utf-8 -*-
"""#8 Self-play / curriculum. Compare RL-blue trained two ways:
   (a) baseline: vs a fixed rule-red (the code-team's cached rl_blue, trained vs rule)
   (b) self-play: alternate training blue vs evolving rl-red and rl-red vs evolving blue
Evaluate both vs {rule-red, rl-red} on held-out seeds. Question: does self-play make the
learned defender more robust against an adaptive (rl) attacker? (Heuristic shown as reference.)
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run
from agents import brains, actions
import agents.rl as rlmod
from agents.rl import QTable, BLUE_Q, RED_Q

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
STEPS = cfg["steps"]


def blue_episode_vs_red(seed, q_blue, eps, rng, q_red):
    """Blue (learning) vs RL-red (frozen greedy). MC-update q_blue from shared blue reward."""
    brains.set_red_q(q_red, 0.0, None, train=False)
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RLRed); n = fleet["n"]; trajs = {}
    for t in range(STEPS):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}
        for a in live:
            s = brains.blue_state(a, ctx); aid = q_blue.act(s, eps, rng)
            acts[a] = actions.make_blue_index(aid, env, a, ctx); picks[a] = (s, aid)
        _, rew, done, _ = env.step(acts); r = float(np.mean(list(rew.values()))) if rew else 0.0
        for a, (s, aid) in picks.items(): trajs.setdefault(a, []).append((s, aid, r))
        if all(done.values()): break
    for tr in trajs.values(): q_blue.learn(tr)


def red_episode_vs_blue(seed, q_red, eps, rng, q_blue):
    """RL-red (learning) vs blue (frozen greedy from q_blue). Reward = newly owned drones."""
    brains.set_red_q(q_red, eps, rng, train=True)
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RLRed); n = fleet["n"]; trajs = {}
    prev = len(run.compromised_drones(cyborg, n))
    for t in range(STEPS):
        comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(q_blue.act(brains.blue_state(a, ctx), 0.0, None), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        stash = brains.pop_red_stash(); now = len(run.compromised_drones(cyborg, n)); r = float(now - prev); prev = now
        for name, s, aid in stash: trajs.setdefault(name, []).append((s, aid, r))
        if all(done.values()): break
    brains.set_red_q(None, 0.0, None, False)
    for tr in trajs.values(): q_red.learn(tr)


def self_play(rounds=4, episodes=60):
    qb = QTable(actions.BLUE_DECISION_N); qr = QTable(actions.RED_N); rng = np.random.default_rng(0)
    for rd in range(rounds):
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.8 * episodes))
            blue_episode_vs_red(rd * episodes + ep, qb, eps, rng, qr)
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.8 * episodes))
            red_episode_vs_blue(rd * episodes + ep, qr, eps, rng, qb)
        print(f"  self-play round {rd+1}/{rounds}  |blueS|={len(qb.q)} |redS|={len(qr.q)}")
    return qb, qr


def eval_blue(blue_fn, seeds, red_brain, red_q=None):
    if red_q is not None:
        brains.set_red_q(red_q, 0.0, None, train=False)
    comps = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, red_brain); n = fleet["n"]
        for t in range(STEPS):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_fn(env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        comps.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(comps))


print("Self-play training (rounds=4) ...")
qb_sp, qr_sp = self_play()
qb_base = QTable.load(BLUE_Q)          # code-team rl-blue, trained vs rule
qr_fixed = QTable.load(RED_Q)          # a fixed adaptive (rl) attacker for testing

def f_heur(env, a, ctx): return brains.blue_decide("rule", env, a, ctx)
def f_base(env, a, ctx): return qb_base.act(brains.blue_state(a, ctx), 0.0, None)
def f_sp(env, a, ctx): return qb_sp.act(brains.blue_state(a, ctx), 0.0, None)

rows = []
print("\n=== final compromise (held-out), lower=better ===")
print(f"{'defender':24}{'vs rule-red':>13}{'vs rl-red':>11}")
for name, fn in [("heuristic", f_heur), ("RL-blue (vs rule)", f_base), ("RL-blue (self-play)", f_sp)]:
    c_rule = eval_blue(fn, EVAL, brains.RuleRed)
    c_rl = eval_blue(fn, EVAL, brains.RLRed, red_q=qr_fixed)
    rows.append((name, c_rule, c_rl))
    print(f"{name:24}{c_rule:13.3f}{c_rl:11.3f}")

labels = [r[0] for r in rows]; vr = [r[1] for r in rows]; vl = [r[2] for r in rows]
x = np.arange(len(labels)); w = 0.36
plt.figure(figsize=(7.4, 4.4))
plt.bar(x - w / 2, vr, w, label="vs rule-red", color="steelblue")
plt.bar(x + w / 2, vl, w, label="vs rl-red (adaptive)", color="indianred")
plt.xticks(x, labels, fontsize=8); plt.ylabel("final compromise (lower=better)")
plt.title("#8 Self-play vs vs-rule training (RL-blue robustness)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig15_selfplay.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_selfplay.csv"), "w", newline="", encoding="utf-8") as f:
    w_ = csv.writer(f); w_.writerow(["defender", "vs_rule", "vs_rl"]); w_.writerows(rows)
print("Saved fig15_selfplay.png, summary_selfplay.csv")
