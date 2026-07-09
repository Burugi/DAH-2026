"""Run the HVT defense against an attack scenario on the DroneSwarm sim.

    python src/run_hvt.py --scenario A1
    python src/run_hvt.py --scenario A7 --recall 0.75 --fp 0.1 --seeds 0 1 2
    python src/run_hvt.py --all

Scenarios come from configs/attack_scenarios.yaml. Each run writes the standard
results/<run_id>/ bundle (log.csv, arrays.npz, meta.json), so the same dashboard,
plot and render tools work on it:

    python src/viz/dashboard.py results/<run_id> --png
"""
import os, sys, time, argparse
import numpy as np
import yaml

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)

import run
from agents import brains
from agents.actions import RED_N
from agents.hvt import HVTDefense
from sim.defense import run_defense
from viz import score
import harness

CONFIG = os.path.join(SRC, "configs", "sweep.yaml")
SCENARIOS = os.path.join(SRC, "configs", "attack_scenarios.yaml")


def load_scenarios():
    raw = yaml.safe_load(open(SCENARIOS, encoding="utf-8"))["scenarios"]
    return {s["id"]: s for s in raw}


def rollout(cfg, spec, seed, recall=1.0, fp=0.0):
    """One HVT episode against `spec`. Returns (fleet, reward, red_owned, defence)."""
    vectors = spec.get("vectors", ["W"])
    tempo = spec.get("tempo", 1.0)
    ml = cfg["fleet"].get("max_link", 40)

    cfg2 = cfg
    if spec.get("start_red"):
        cfg2 = dict(cfg)
        cfg2["sim"] = dict(cfg["sim"])
        cfg2["sim"]["starting_num_red"] = spec["start_red"]

    fleet, cyborg, env, ip2d = run.build_env(cfg2, seed, harness.make_red(vectors, tempo))
    n = fleet["n"]
    rng = np.random.default_rng(seed + 4)

    hubs, black = harness.scenario_hubs_black(fleet, spec, ml, rng)
    policy = HVTDefense()
    policy.reset(cfg, fleet, spec, hubs, black, ml, recall, fp)

    reward = np.zeros(cfg["steps"])
    red_owned = np.zeros((cfg["steps"], n), np.int8)
    red_log, blue_log = [], []
    brains.pop_red_actlog()

    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts, _ = policy.step(comp, pos, env, live, ip2d, rng)
        blue_log.append(harness.blue_action_tape(env, acts))
        _, rew, done, _ = env.step(acts)
        red_log.append(run._summ(brains.pop_red_actlog(), RED_N))
        reward[t] = float(np.mean(list(rew.values()))) if rew else 0.0
        for d in run.compromised_drones(cyborg, n):
            red_owned[t, d] = 1
        if all(done.values()):
            reward, red_owned = reward[:t + 1], red_owned[:t + 1]
            from sim.fleet import truncate
            truncate(fleet, t + 1)
            break

    T = len(reward)
    a_t, d_t = score.per_step(red_owned, fleet["link_up"][:T])
    defence = run_defense(cfg, fleet, seed, red_owned)
    from agents.actions import BLUE_DECISION_N
    defence.update(
        red_act=np.array([r[0] for r in red_log], np.int16),
        blue_act=np.array([b[0] for b in blue_log], np.int16),
        red_cnt=np.stack([r[1] for r in red_log]) if red_log else np.zeros((T, RED_N), np.int16),
        blue_cnt=np.stack([b[1] for b in blue_log]) if blue_log else np.zeros((T, BLUE_DECISION_N), np.int16),
        a_t=a_t, d_t=d_t)
    return fleet, reward, red_owned, defence


def run_scenario(scenarios, sid, seeds, recall, fp):
    spec = scenarios[sid]
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    cfg = dict(cfg)
    cfg["name"] = f"{sid} {spec.get('name', '')}".strip()
    red_label = "+".join(spec.get("vectors", ["W"]))

    tag = f"_r{recall}_fp{fp}" if (recall < 1.0 or fp > 0.0) else ""
    run_id = f"hvt_{sid}{tag}"
    out = os.path.join(RESULTS, run_id)

    print(f"HVT  {cfg['name']}  vectors={red_label} "
          f"detector_q={spec.get('detector_q', 1.0)} recall={recall} fp={fp}")
    t0, results = time.time(), {}
    for seed in seeds:
        fleet, reward, red_owned, df = rollout(cfg, spec, seed, recall, fp)
        results[seed] = (fleet, reward, red_owned, df)
        print(f"  seed {seed}: compromised={int(red_owned[-1].sum())}/{fleet['n']} "
              f"peak={int(red_owned.sum(1).max())}")

    metrics = run.save_run(cfg, out, red_label, "hvt", results)
    print(f"-> results/{run_id}/  ({round(time.time() - t0, 1)}s)  "
          f"final_comp={metrics['final_compromise']} availability={metrics['availability']} "
          f"defense={metrics['defense_score']}")
    print(f"   dashboard: python src/viz/dashboard.py results/{run_id} --png")
    return run_id, metrics


def main():
    ap = argparse.ArgumentParser(description="Run the HVT defense on an attack scenario.")
    ap.add_argument("--scenario", metavar="ID", help="scenario id (e.g. A1, A7, A17)")
    ap.add_argument("--all", action="store_true", help="run every scenario")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--recall", type=float, default=1.0, help="detection recall (<1 = misses)")
    ap.add_argument("--fp", type=float, default=0.0, help="false-positive rate")
    a = ap.parse_args()

    scenarios = load_scenarios()
    if a.all:
        for sid in scenarios:
            run_scenario(scenarios, sid, a.seeds, a.recall, a.fp)
    elif a.scenario:
        if a.scenario not in scenarios:
            ap.error(f"unknown scenario {a.scenario}; choices: {', '.join(scenarios)}")
        run_scenario(scenarios, a.scenario, a.seeds, a.recall, a.fp)
    else:
        ap.error("give --scenario ID or --all")


if __name__ == "__main__":
    main()
