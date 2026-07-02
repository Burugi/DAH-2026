# -*- coding: utf-8 -*-
"""Agentic-system efficiency metrics: ACTION EFFICIENCY + DECISION LATENCY.
  Action efficiency : # intervention actions (Retake/Remove/Block/Decoy) used per episode, and
                      containment achieved per intervention. Fewer interventions for the same
                      containment = cheaper on availability (the scoring multiplier).
  Decision latency  : wall-clock time of the blue decision function per drone-decision (us). The
                      sim step is excluded; this is the agent's own compute -> embedded real-time
                      feasibility. (Measured on this host; relative ordering is what matters.)
"""
import sys, os, time
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [3000, 3001, 3002, 3003, 3004]
INTERVENTION = {3, 4, 5, 6, 8}        # state-changing defensive ops (availability-costing)


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(kind, i, comp, pos, ml, rising):
    if i in comp: return 3
    fr = frontier(i, comp, pos, ml)
    if kind == "flat":
        return 4 if comp else 1
    if kind == "pred":
        return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "predOODA":
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    if kind == "adaptive":
        if rising: return 6 if fr else (4 if comp else 1)
        return 4 if comp else 7
    return 1


def rollout(seed, defense):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RuleRed); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    hist = []; interventions = 0; dec_time = 0.0; dec_count = 0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        t0 = time.perf_counter()
        for a in live:
            i = int(a.split("_")[-1])
            aid = blue_act(defense, i, comp, pos, ml, rising)          # policy decision
            idx = actions.make_blue_index(aid, env, a, ctx)            # map to wrapper index
            acts[a] = idx
            if aid in INTERVENTION:
                interventions += 1
            dec_count += 1
        dec_time += time.perf_counter() - t0
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    comp_final = len(run.compromised_drones(cyborg, n)) / n
    us_per_dec = (dec_time / max(1, dec_count)) * 1e6
    return comp_final, interventions, us_per_dec


DEFENSES = ["flat", "pred", "predOODA", "adaptive"]
print("=== action efficiency + decision latency (vs RuleRed, 5 seeds) ===")
print("defense".ljust(11) + "점령    개입행동/ep   점령감소/개입   지연 us/결정")
rows = []
nodef_comp = 0.889   # from exp_metrics (no-defense rule-worm floor)
for d in DEFENSES:
    cs, iv, us = [], [], []
    for s in EVAL:
        c, n_iv, u = rollout(s, d); cs.append(c); iv.append(n_iv); us.append(u)
    c_m, iv_m, us_m = np.mean(cs), np.mean(iv), np.mean(us)
    reduced = max(0.0, nodef_comp - c_m)
    eff = reduced / iv_m if iv_m > 0 else 0.0          # containment per intervention
    rows.append((d, c_m, iv_m, eff, us_m))
    print(f"{d:11}{c_m:.3f}   {iv_m:8.1f}    {eff:11.4f}      {us_m:8.1f}")

with open(os.path.join(OUT, "summary_cost_latency.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["defense", "점령", "개입행동_per_ep", "점령감소_per_개입", "지연_us_per_결정"])
    for d, c, iv, eff, us in rows: wr.writerow([d, round(c, 3), round(iv, 1), round(eff, 4), round(us, 1)])

fastest = min(rows, key=lambda r: r[4]); leanest = min(rows, key=lambda r: r[2])
print(f"\n최저지연: {fastest[0]} {fastest[4]:.1f} us/결정 (sub-ms -> 실시간 가능)")
print(f"최소개입(가용성 친화): {leanest[0]} {leanest[2]:.1f} 개입/ep")
print("Saved summary_cost_latency.csv")
