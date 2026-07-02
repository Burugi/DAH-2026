# -*- coding: utf-8 -*-
"""Proof that the sims are real: print CybORG version + live worm spread per step."""
import sys, os, yaml
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import run
from agents import brains, actions
from CybORG import CYBORG_VERSION

print("CybORG version:", CYBORG_VERSION)
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
fleet, cyborg, env, ip2d = run.build_env(cfg, 1000, brains.RuleRed)
n = fleet["n"]
print(f"env type: {type(env).__name__}, cyborg: {type(cyborg).__name__}, drones: {n}")
print("LIVE worm sim — compromised drone IDs per step (rule-red vs rule-blue):")
for t in range(10):
    comp = run.compromised_drones(cyborg, n)
    ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
    live = [a for a in env.active_agents if a in env.agent_actions]
    acts = {a: actions.make_blue_index(brains.blue_decide("rule", env, a, ctx), env, a, ctx) for a in live}
    print(f"  step {t:2d}: {len(comp)}/{n} compromised  ids={sorted(comp)}")
    env.step(acts)
