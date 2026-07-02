# -*- coding: utf-8 -*-
"""Real fix for size overfit: multi-size CURRICULUM training of HYBRID+.
HYBRID+ overfit because it trained on one cluster size (18). Here we train the SAME
richer-state hybrid but each episode draws a random size from {12,18,30}. Then evaluate
across sizes vs single-size HYBRID+ and heuristic. Does the curriculum fix generalization?
"""
import sys, os, copy, random
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import run
from agents import brains, actions
from agents.rl import QTable
np.random.seed(0); random.seed(0)

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
BASE["attacks"] = []
EVAL = [1000, 1001, 1002, 1003, 1004]
PRO = [4, 6, 8]; MU = 0.3
SIZES = [(8, 4), (12, 6), (20, 10)]      # 12 / 18 / 30


def inf_nbr(i, comp, pos, ml):
    return sum(1 for d in comp if d != i and np.linalg.norm(pos[i] - pos[d]) < ml)

def rstate(i, comp, pos, n, ml):
    return (1 if i in comp else 0, min(3, inf_nbr(i, comp, pos, ml)), min(3, int(len(comp) / n * 4)))


def sized_cfg(nu, ng):
    c = copy.deepcopy(BASE); c["fleet"]["n_uav"] = nu; c["fleet"]["n_ugv"] = ng; return c


def train_hp(episodes=300, curriculum=False):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        nu, ng = (SIZES[ep % len(SIZES)] if curriculum else (12, 6))   # base train size = 18 (12,6)
        cfg = sized_cfg(nu, ng)
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed)
        n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}; blk = 0
            for a in live:
                i = int(a.split("_")[-1]); s = rstate(i, comp, pos, n, ml)
                if i in comp: aid = 3
                else:
                    idx = q.act(s, eps, rng); aid = PRO[idx]; picks[a] = (s, idx)
                if aid in (6, 8): blk += 1
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            r = (float(np.mean(list(rew.values()))) if rew else 0.0) - MU * blk
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, r))
            if all(done.values()): break
        for tr in trajs.values(): q.learn(tr)
    return q


def eval_decide(cfg, decide, seeds):
    cs = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, brains.RuleRed); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(decide(int(a.split("_")[-1]), comp, pos, n, ml), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


print("train HYBRID+ (single size 18) ..."); q_single = train_hp(curriculum=False)
print("train HYBRID+ (multi-size curriculum 12/18/30) ..."); q_curr = train_hp(curriculum=True)

def d_heur(i, comp, pos, n, ml): return 3 if i in comp else (4 if comp else 1)
def d_single(i, comp, pos, n, ml): return 3 if i in comp else PRO[q_single.act(rstate(i, comp, pos, n, ml), 0.0, None)]
def d_curr(i, comp, pos, n, ml): return 3 if i in comp else PRO[q_curr.act(rstate(i, comp, pos, n, ml), 0.0, None)]

res = {"휴리스틱": [], "HYBRID+ (단일크기)": [], "HYBRID+ (커리큘럼)": []}
labs = [nu + ng for nu, ng in SIZES]
print("\n=== size generalization (vs rule-red) ===")
print("size   휴리스틱   단일크기   커리큘럼")
for nu, ng in SIZES:
    cfg = sized_cfg(nu, ng)
    h = eval_decide(cfg, d_heur, EVAL); s1 = eval_decide(cfg, d_single, EVAL); sc = eval_decide(cfg, d_curr, EVAL)
    res["휴리스틱"].append(h); res["HYBRID+ (단일크기)"].append(s1); res["HYBRID+ (커리큘럼)"].append(sc)
    print(f"{nu+ng:>4}    {h:.3f}    {s1:.3f}    {sc:.3f}")

plt.figure(figsize=(7.4, 4.6))
for k, col, mk in [("휴리스틱", "seagreen", "o"), ("HYBRID+ (단일크기)", "purple", "s"), ("HYBRID+ (커리큘럼)", "crimson", "^")]:
    plt.plot(labs, res[k], mk + "-", color=col, lw=2, label=k)
plt.xlabel("군집 크기"); plt.ylabel("최종 점령 (낮을수록 좋음)")
plt.title("한계돌파2: 다중크기 커리큘럼이 과적합을 해결하나")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig26_curriculum.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_curriculum.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["size"] + list(res.keys()))
    for i, s in enumerate(labs): w.writerow([s] + [round(res[k][i], 3) for k in res])
print("\nSaved fig26_curriculum.png, summary_curriculum.csv")
print("RESULT:", {k: [round(v, 3) for v in vs] for k, vs in res.items()})
