# -*- coding: utf-8 -*-
"""Extra empirical experiments on the code-team harness:
 #1 unseen-attack generalization, #2 attack-intensity sensitivity + 95% CI,
 #3 hybrid tuning (proactive set) / #4 decoy isolation, #5 availability self-harm,
 #7 swarm-size generalization. Each wrapped in try/except; saves figs + a results log.
"""
import sys, os, copy, csv, traceback
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run
from agents import brains, actions
import agents.rl as rlmod
from agents.rl import QTable, BLUE_Q

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
SEEDS20 = list(range(1000, 1020))
log = open(os.path.join(OUT, "exp_extra_results.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

q_multi = QTable.load(BLUE_Q)
try:
    q_red = QTable.load(rlmod.RED_Q)
except Exception:
    q_red = None
brains.use_rl(q_multi, q_red)   # installs blue + red Q for greedy eval


def eval_blue(cfg, blue_fn, seeds, red=brains.RuleRed):
    comps, rews = [], []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, red)
        n = fleet["n"]; tot = 0.0
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_fn(env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            tot += float(np.mean(list(rew.values()))) if rew else 0.0
            if all(done.values()):
                break
        comps.append(len(run.compromised_drones(cyborg, n)) / n); rews.append(tot)
    return np.array(comps), np.array(rews)

def ci95(a): return 1.96 * np.std(a) / np.sqrt(len(a))

def train_hybrid(cfg, PRO, episodes=300):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed); n = fleet["n"]; trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n)
            ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts, picks = {}, {}
            for a in live:
                own = int(a.split("_")[-1])
                if own in comp:
                    aid = 3
                else:
                    s = brains.blue_state(a, ctx); idx = q.act(s, eps, rng); aid = PRO[idx]; picks[a] = (s, idx)
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts); r = float(np.mean(list(rew.values()))) if rew else 0.0
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, r))
            if all(done.values()): break
        for tr in trajs.values(): q.learn(tr)
    return q

def hybrid_fn(q, PRO):
    def fn(env, a, ctx):
        own = int(a.split("_")[-1])
        if own in ctx["compromised"]: return 3
        return PRO[q.act(brains.blue_state(a, ctx), 0.0, None)]
    return fn

def blue_heur(env, a, ctx): return brains.blue_decide("rule", env, a, ctx)
def blue_rlmulti(env, a, ctx): return q_multi.act(brains.blue_state(a, ctx), 0.0, None)

P("Training base hybrid (PROACTIVE=[4,6,8]) ...")
Q_HYB = train_hybrid(BASE, [4, 6, 8]); HYB = hybrid_fn(Q_HYB, [4, 6, 8])

# ============ #1 unseen-attack generalization ============
try:
    P("\n===== #1 UNSEEN-ATTACK GENERALIZATION (hybrid trained vs rule-red only) =====")
    reds = {"rule(seen)": brains.RuleRed, "rl(unseen)": brains.RLRed, "llm(unseen)": brains.LLMRed}
    P(f"{'defender':22}" + "".join(f"{r:>14}" for r in reds))
    g1 = {}
    for dname, dfn in [("heuristic", blue_heur), ("hybrid(vs rule)", HYB), ("RL-multi", blue_rlmulti)]:
        row = []
        for rname, rb in reds.items():
            c, _ = eval_blue(BASE, dfn, EVAL, red=rb); row.append(c.mean())
        g1[dname] = row
        P(f"{dname:22}" + "".join(f"{v:14.3f}" for v in row))
    P("해석: hybrid가 학습 안 한 rl/llm 공격에도 점령을 낮게 유지하면 일반화 성공.")
except Exception:
    P("#1 FAILED\n" + traceback.format_exc())

# ============ #2 attack-intensity sensitivity + CI ============
try:
    P("\n===== #2 INTENSITY SENSITIVITY (spawn_rate sweep, 20 seeds, mean +/- 95%CI) =====")
    spawns = [0.05, 0.10, 0.20, 0.35]
    series = {"heuristic": [], "hybrid": [], "RL-multi": []}
    for sp in spawns:
        cfg = copy.deepcopy(BASE); cfg["sim"]["red_spawn_rate"] = sp
        for dname, dfn in [("heuristic", blue_heur), ("hybrid", HYB), ("RL-multi", blue_rlmulti)]:
            c, _ = eval_blue(cfg, dfn, SEEDS20); series[dname].append((c.mean(), ci95(c)))
            P(f"  spawn={sp:.2f}  {dname:10} comp={c.mean():.3f} +/-{ci95(c):.3f}")
    plt.figure(figsize=(7, 4.4))
    for dname, col in [("heuristic", "seagreen"), ("hybrid", "royalblue"), ("RL-multi", "indianred")]:
        m = [x[0] for x in series[dname]]; e = [x[1] for x in series[dname]]
        plt.errorbar(spawns, m, yerr=e, marker="o", capsize=4, label=dname, color=col)
    plt.xlabel("worm spawn rate (attack intensity)"); plt.ylabel("final compromised fraction")
    plt.title("#2 Sensitivity: defenders vs attack intensity (20 seeds, 95% CI)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig12_sensitivity.png"), dpi=130); plt.close()
    P("-> fig12_sensitivity.png")
except Exception:
    P("#2 FAILED\n" + traceback.format_exc())

# ============ #3 hybrid tuning / #4 decoy isolation ============
try:
    P("\n===== #3/#4 HYBRID PROACTIVE-SET TUNING (decoy/block isolation) =====")
    P(f"{'PROACTIVE set':22}{'comp':>8}{'reward':>9}")
    tune = {}
    for PRO in [[4], [4, 6], [4, 8], [4, 6, 8]]:
        q = train_hybrid(BASE, PRO); fn = hybrid_fn(q, PRO)
        c, r = eval_blue(BASE, fn, EVAL); tune[str(PRO)] = (c.mean(), r.mean())
        names = {4: "retake", 6: "block", 8: "decoy"}
        P(f"{'+'.join(names[i] for i in PRO):22}{c.mean():8.3f}{r.mean():9.0f}")
    P("해석: retake만[4] 대비 +block/+decoy의 점령·보상 변화 = 각 행동의 순효과.")
except Exception:
    P("#3/#4 FAILED\n" + traceback.format_exc())

# ============ #5 availability self-harm ============
try:
    P("\n===== #5 AVAILABILITY SELF-HARM (response aggressiveness vs reward) =====")
    def make_level(L):
        def fn(env, a, ctx):
            own = int(a.split("_")[-1]); comp = ctx["compromised"]
            if own in comp: return 3                       # always clean own (except L0)
            if L == 0: return 1                            # monitor only
            if L == 1: return 1                            # remove-own only
            if L == 2: return 4 if comp else 1             # +retake (=rule)
            if L == 3: return 6 if comp else 1             # +block compromised (aggressive isolate)
            return 6                                       # L4: block always (over-response)
        return fn
    lv = {"L0 monitor": 0, "L1 remove-own": 1, "L2 rule(retake)": 2, "L3 block-comp": 3, "L4 block-all": 4}
    comp_l, rew_l, labs = [], [], []
    for name, L in lv.items():
        c, r = eval_blue(BASE, make_level(L), EVAL); comp_l.append(c.mean()); rew_l.append(r.mean()); labs.append(name)
        P(f"  {name:18} comp={c.mean():.3f} reward={r.mean():.0f}")
    fig, ax1 = plt.subplots(figsize=(7.6, 4.4)); x = np.arange(len(labs))
    ax1.bar(x, comp_l, color="indianred", alpha=0.8); ax1.set_ylabel("final compromise", color="indianred")
    ax1.set_xticks(x); ax1.set_xticklabels(labs, fontsize=8, rotation=10)
    ax2 = ax1.twinx(); ax2.plot(x, rew_l, "o-", color="navy"); ax2.set_ylabel("reward (availability proxy)", color="navy")
    plt.title("#5 Over-response self-harm: aggressive isolation tanks reward")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig13_availability.png"), dpi=130); plt.close()
    P("-> fig13_availability.png  (L3/L4 block은 점령은 비슷하나 보상 급락이면 자해 입증)")
except Exception:
    P("#5 FAILED\n" + traceback.format_exc())

# ============ #7 swarm-size generalization ============
try:
    P("\n===== #7 SWARM-SIZE GENERALIZATION (per-agent policies, no synthetic attacks) =====")
    sizes = [(8, 4), (12, 6), (20, 10)]
    P(f"{'defender':14}" + "".join(f"{nu+ng:>10}" for nu, ng in sizes))
    for dname, dfn in [("heuristic", blue_heur), ("hybrid", HYB), ("RL-multi", blue_rlmulti)]:
        row = []
        for nu, ng in sizes:
            cfg = copy.deepcopy(BASE); cfg["fleet"]["n_uav"] = nu; cfg["fleet"]["n_ugv"] = ng; cfg["attacks"] = []
            c, _ = eval_blue(cfg, dfn, EVAL); row.append(c.mean())
        P(f"{dname:14}" + "".join(f"{v:10.3f}" for v in row))
    P("해석: 드론별 정책은 군집 크기(12/18/30)가 달라도 점령 비율이 비슷하면 크기 일반화 성공.")
except Exception:
    P("#7 FAILED\n" + traceback.format_exc())

P("\nDONE exp_extra.")
log.close()
