"""Run one matchup: a red agent type vs a blue agent type on the DroneSwarm sim.

    python run.py configs/sweep.yaml --red rule --blue rule
    python run.py configs/sweep.yaml --red llm --blue rl --scenario A1

red/blue type each in {rule, llm, rl} (see brains.py). Writes results/<run_id>/
with log.csv (long format), arrays.npz and meta.json. The 3x3 comparison lives
in sweep.py, which calls rollout() below for all nine matchups.
"""
import os, sys, json, csv, time, hashlib, argparse
import numpy as np
import yaml

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)
from sim.fleet import generate_fleet, starting_positions, truncate
from sim.defense import run_defense
from agents.actions import make_blue_index, RED_N, BLUE_DECISION_N, RED_CATALOG, BLUE_CATALOG
from agents import brains
from agents.brains import RED_BRAINS, blue_decide, use_rl
from viz import score

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper

COLS = ["run_id", "scenario", "red_type", "blue_type", "seed", "step",
        "entity_id", "entity_type", "pos_true_x", "pos_true_y", "pos_rep_x", "pos_rep_y",
        "gps_err", "gps_err_corr", "snr", "link_up", "red_owned",
        "reward_step_mean", "atk_jam", "atk_gps", "atk_exploit",
        "det_jam", "det_gps", "isolated"]


def compromised_drones(cyborg, n):
    state = cyborg.environment_controller.state
    owned = set()
    for agent, sessions in state.sessions.items():
        if "red" in agent.lower():
            for s in sessions.values():
                host = getattr(s, "hostname", "")
                if host.startswith("drone_") and int(host.split("_")[-1]) < n:
                    owned.add(int(host.split("_")[-1]))
    return owned


def build_env(cfg, seed, red_class):
    """Construct the synthetic fleet + DroneSwarm sim for `red_class` as red."""
    fl, sim = cfg["fleet"], cfg["sim"]
    fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], seed,
                           fl["grid"], cfg.get("attacks"), fl.get("max_link", 40))
    sg = DroneSwarmScenarioGenerator(
        num_drones=fleet["n"], maximum_steps=cfg["steps"],
        default_red_agent=red_class,
        starting_num_red=sim.get("starting_num_red", 1),
        red_spawn_rate=sim.get("red_spawn_rate", 0.05),
        max_length_data_links=sim.get("max_length_data_links", 40),
        starting_positions=starting_positions(fleet))
    cyborg = CybORG(sg, "sim", seed=int(seed))
    env = PettingZooParallelWrapper(env=cyborg)
    env.reset()
    ip_to_drone = {ip: int(h.split("_")[-1]) for h, ip in cyborg.get_ip_map().items()
                   if h.startswith("drone_")}
    return fleet, cyborg, env, ip_to_drone


def _worm_step(t, cfg, cyborg, ip_to_drone, owned, rng):
    """Advance worm propagation by one step if the scenario defines a worm block.

    Reads cfg['worm'] and, after dormancy_steps have passed, exploits the
    highest-SNR neighbour of each infected drone every spread_interval steps.
    Uses CybORG's internal state directly (no env.step call) so it does not
    consume a blue action slot.
    """
    w = cfg.get("worm")
    if not w:
        return
    if t < w.get("dormancy_steps", 0):
        return
    if (t - w.get("dormancy_steps", 0)) % w.get("spread_interval", 4) != 0:
        return
    if len(owned) >= w.get("max_infected", 999):
        return

    drone_to_ip = {v: k for k, v in ip_to_drone.items()}
    infected = list(owned)
    mode = w.get("spread_mode", "proximity")

    for src_id in infected:
        if len(owned) >= w.get("max_infected", 999):
            break
        src_ip = drone_to_ip.get(src_id)
        if src_ip is None:
            continue
        candidates = [d for d in range(len(ip_to_drone)) if d not in owned
                      and d in drone_to_ip]
        if not candidates:
            continue
        if mode == "proximity":
            target_id = rng.choice(candidates)   # proximity best-effort via random
        elif mode == "sequential":
            target_id = min(candidates)
        else:
            target_id = rng.choice(candidates)
        tgt_ip = drone_to_ip.get(target_id)
        if tgt_ip is None:
            continue
        try:
            from CybORG.Simulator.Actions import ExploitDroneVulnerability, SeizeControl
            red_agent = next(a for a in cyborg.environment_controller.agent_interfaces
                             if "red" in a.lower())
            session = 0
            exploit = ExploitDroneVulnerability(
                ip_address=tgt_ip, agent=red_agent, session=session)
            result = cyborg.environment_controller.execute_action(
                exploit, red_agent)
            if getattr(result, "success", False) is not False:
                seize = SeizeControl(
                    ip_address=tgt_ip, agent=red_agent, session=session)
                cyborg.environment_controller.execute_action(seize, red_agent)
        except Exception:
            pass   # worm spread is best-effort; never crash the rollout


def _summ(aids, k):
    """(rep_action_id, count_vector[k]) for the action ids chosen this step."""
    cnt = np.bincount([a for a in aids if 0 <= a < k], minlength=k).astype(np.int16)
    rep = int(cnt.argmax()) if cnt.sum() else 0
    return rep, cnt


def rollout(cfg, seed, red_type="rule", blue_type="rule"):
    """One evaluation episode. RL policies must be installed via use_rl() first."""
    fleet, cyborg, env, ip_to_drone = build_env(cfg, seed, RED_BRAINS[red_type])
    n = fleet["n"]
    import random; random.seed(seed)
    rng_worm = np.random.default_rng(seed + 42)
    reward = np.zeros(cfg["steps"])
    red_owned = np.zeros((cfg["steps"], n), np.int8)
    red_log, blue_log = [], []           # per-step (rep_id, count_vector) for the dashboard
    brains.pop_red_actlog()              # clear any leftovers from env.reset()

    # ── Pre-compromise: insider / side-channel / leader-takeover (A12/A15/A17) ──
    scenario = cfg.get("_scenario") or {}
    pre_comp = scenario.get("pre_compromise", {})
    if pre_comp:
        drone_to_ip = {v: k for k, v in ip_to_drone.items()}
        for drone_id in pre_comp.get("drones", []):
            tgt_ip = drone_to_ip.get(drone_id)
            if tgt_ip is None:
                continue
            try:
                from CybORG.Simulator.Actions import ExploitDroneVulnerability, SeizeControl
                red_agent = next(a for a in cyborg.environment_controller.agent_interfaces
                                 if "red" in a.lower())
                exploit = ExploitDroneVulnerability(
                    ip_address=tgt_ip, agent=red_agent, session=0)
                cyborg.environment_controller.execute_action(exploit, red_agent)
                seize = SeizeControl(ip_address=tgt_ip, agent=red_agent, session=0)
                cyborg.environment_controller.execute_action(seize, red_agent)
            except Exception:
                pass

    for t in range(cfg["steps"]):
        owned = compromised_drones(cyborg, n)

        # ── Worm propagation engine (A1, A9, A14, A17) ──────────────────────
        _worm_step(t, cfg, cyborg, ip_to_drone, owned, rng_worm)

        ctx = {"compromised": owned, "ip_to_drone": ip_to_drone, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        blue_aids = [blue_decide(blue_type, env, a, ctx) for a in live]
        acts = {a: make_blue_index(aid, env, a, ctx) for a, aid in zip(live, blue_aids)}
        _, rew, done, _ = env.step(acts)
        blue_log.append(_summ(blue_aids, BLUE_DECISION_N))
        red_log.append(_summ(brains.pop_red_actlog(), RED_N))
        reward[t] = float(np.mean(list(rew.values()))) if rew else 0.0
        for d in compromised_drones(cyborg, n):
            red_owned[t, d] = 1
        if all(done.values()):
            reward, red_owned = reward[:t + 1], red_owned[:t + 1]
            truncate(fleet, t + 1)
            break
    T = len(reward)
    a_t, d_t = score.per_step(red_owned, fleet["link_up"][:T])
    defence = run_defense(cfg, fleet, seed, red_owned)
    defence.update(
        red_act=np.array([r[0] for r in red_log], np.int16),
        blue_act=np.array([b[0] for b in blue_log], np.int16),
        red_cnt=np.stack([r[1] for r in red_log]) if red_log else np.zeros((T, RED_N), np.int16),
        blue_cnt=np.stack([b[1] for b in blue_log]) if blue_log else np.zeros((T, BLUE_DECISION_N), np.int16),
        a_t=a_t, d_t=d_t)
    return fleet, reward, red_owned, defence


def attack_defense_metrics(reward, red_owned, n, dmetric):
    """Quantitative attack + defence scores for one episode."""
    comp_frac = red_owned.sum(1) / n                       # per step
    first = np.argmax(comp_frac > 0) if comp_frac.any() else len(comp_frac)
    drops = np.clip(-np.diff(red_owned.sum(1)), 0, None).sum()   # retaken count
    return {
        "final_compromise": float(comp_frac[-1]),
        "peak_compromise": float(comp_frac.max()),
        "time_to_first_compromise": int(first),
        "compromise_auc": float(comp_frac.mean()),
        "blue_reward_total": float(reward.sum()),
        "recovered": int(drops),
        "comp_F1":        dmetric["comp"]["F1"],
        "jam_F1":         dmetric["jam"]["F1"],
        "gps_F1":         dmetric["gps"]["F1"],
        "link_drop_F1":   dmetric.get("link_drop",  {}).get("F1", 0.0),
        "snr_poison_F1":  dmetric.get("snr_poison", {}).get("F1", 0.0),
        "bw_drain_F1":    dmetric.get("bw_drain",   {}).get("F1", 0.0),
        "gps_err_before": dmetric["gps_err_before"],
        "gps_err_after":  dmetric["gps_err_after"],
    }


def save_run(cfg, out, red_type, blue_type, results):
    """Write log.csv + arrays.npz + meta.json for a finished matchup into `out`."""
    os.makedirs(out, exist_ok=True)
    n_uav, n = cfg["fleet"]["n_uav"], cfg["fleet"]["n_uav"] + cfg["fleet"]["n_ugv"]
    types = ["uav"] * n_uav + ["ugv"] * (n - n_uav)
    dfn = cfg.get("defense") or {}
    run_id = os.path.basename(out)
    rows, stacks, dmetrics, emetrics = [], {}, [], []

    for seed, (fleet, reward, red_owned, df) in results.items():
        m = df["metrics"]; dmetrics.append(m)
        emetrics.append(attack_defense_metrics(reward, red_owned, n, m))
        T = len(reward)
        seed_arrays = {"reward": reward, "red_owned": red_owned, "gps_corr": df["gps_corr"],
                       "det_jam": df["det_jam"], "det_gps": df["det_gps"], "det_comp": df["det_comp"],
                       "isolated": df["isolated"],
                       "red_act": df.get("red_act", np.zeros(T, np.int16)),
                       "blue_act": df.get("blue_act", np.zeros(T, np.int16)),
                       "red_cnt": df.get("red_cnt", np.zeros((T, RED_N), np.int16)),
                       "blue_cnt": df.get("blue_cnt", np.zeros((T, BLUE_DECISION_N), np.int16)),
                       "a_t": df.get("a_t", np.zeros(T)), "d_t": df.get("d_t", np.zeros(T)),
                       **{k: fleet[k] for k in ("snr", "gps_err", "link_up", "label_jam",
                                                "label_gps", "pos_true", "pos_rep")}}
        for k, v in seed_arrays.items():
            stacks.setdefault(k, []).append(np.asarray(v))
        for t in range(T):
            for e in range(n):
                rows.append([run_id, cfg["name"], red_type, blue_type, seed, t, e, types[e],
                             *np.round(fleet["pos_true"][t, e], 2), *np.round(fleet["pos_rep"][t, e], 2),
                             round(float(fleet["gps_err"][t, e]), 2), round(float(df["gps_corr"][t, e]), 2),
                             round(float(fleet["snr"][t, e]), 2), int(fleet["link_up"][t, e]),
                             int(red_owned[t, e]), round(float(reward[t]), 3),
                             int(fleet["label_jam"][t, e]), int(fleet["label_gps"][t, e]), int(red_owned[t, e]),
                             int(df["det_jam"][t, e]), int(df["det_gps"][t, e]), int(df["isolated"][t, e])])

    Tmin = min(len(s) for s in stacks["reward"])
    npz = {k: np.stack([a[:Tmin] for a in v]) for k, v in stacks.items()}
    npz["types"] = np.array(types)
    np.savez(os.path.join(out, "arrays.npz"), **npz)
    with open(os.path.join(out, "log.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([COLS] + rows)

    avg = lambda key: round(float(np.mean([e[key] for e in emetrics])), 3)
    std = lambda key: round(float(np.std([e[key] for e in emetrics])), 3)
    metrics = {k: avg(k) for k in emetrics[0]}
    metrics_std = {k: std(k) for k in emetrics[0]}
    metrics.update(score.episode_scores(metrics, npz["red_owned"], npz["link_up"], int(Tmin)))
    json.dump({"run_id": run_id, "cyborg": CYBORG_VERSION, "config": cfg,
               "red_type": red_type, "blue_type": blue_type,
               "n_entities": n, "types": types, "seeds": list(results),
               "steps_used": int(Tmin),
               "red_action_names": [c[0] for c in RED_CATALOG],
               "blue_action_names": [c[0] for c in BLUE_CATALOG[:BLUE_DECISION_N]],
               "defense": {"detector": dfn.get("detector", "none"),
                           "response": dfn.get("response", "none"),
                           "comp_F1":       metrics["comp_F1"],
                           "jam_F1":        metrics["jam_F1"],
                           "gps_F1":        metrics["gps_F1"],
                           "link_drop_F1":  metrics["link_drop_F1"],
                           "snr_poison_F1": metrics["snr_poison_F1"],
                           "bw_drain_F1":   metrics["bw_drain_F1"],
                           "gps_err_before": metrics["gps_err_before"],
                           "gps_err_after":  metrics["gps_err_after"]},
               "metrics": metrics, "metrics_std": metrics_std},
              open(os.path.join(out, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return metrics


def _apply_scenario(cfg, scenario_id):
    """Load a scenario YAML and merge its attacks/defense into cfg. Returns updated cfg."""
    from scenarios import load_scenario
    return load_scenario(scenario_id, cfg)


def main():
    ap = argparse.ArgumentParser(
        description="Run one red-vs-blue matchup on the DroneSwarm sim.")
    ap.add_argument("config")
    ap.add_argument("--red", default="rule", choices=list(RED_BRAINS))
    ap.add_argument("--blue", default="rule", choices=["rule", "llm", "rl"])
    ap.add_argument("--scenario", default=None, metavar="ID",
                    help="attack scenario id to inject (e.g. A1, A7, A14)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))

    if a.scenario:
        cfg = _apply_scenario(cfg, a.scenario)
        print(f"scenario: {a.scenario} — attacks: "
              f"{[atk['type'] for atk in cfg.get('attacks') or []]}")

    if "rl" in (a.red, a.blue):                            # eval needs frozen policies
        from agents.rl import ensure_trained
        use_rl(*ensure_trained(cfg))

    scenario_tag = f"_{a.scenario}" if a.scenario else ""
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_id = f"{cfg['name']}_{a.red}_vs_{a.blue}{scenario_tag}_{h}"
    out = os.path.join(RESULTS, run_id)
    dfn = cfg.get("defense") or {}
    print(f"CybORG v{CYBORG_VERSION}  {cfg['name']}  red={a.red} blue={a.blue} "
          f"defense={dfn.get('detector', 'none')}/{dfn.get('response', 'none')}")

    t0, results = time.time(), {}
    for seed in cfg["seeds"]:
        fleet, reward, red_owned, df = rollout(cfg, seed, a.red, a.blue)
        results[seed] = (fleet, reward, red_owned, df)
        m = df["metrics"]
        print(f"  seed {seed}: totR={reward.sum():.0f} compromised={int(red_owned[-1].sum())}/{fleet['n']} "
              f"compF1={m['comp']['F1']} jamF1={m['jam']['F1']} gpsF1={m['gps']['F1']}")

    metrics = save_run(cfg, out, a.red, a.blue, results)
    print(f"-> results/{run_id}/  ({round(time.time() - t0, 1)}s)  "
          f"final_comp={metrics['final_compromise']} blueR={metrics['blue_reward_total']}")
    print(f"   plot:   python src/viz/plot.py {run_id}")
    print(f"   render: python src/viz/render.py {run_id}        (live; add --gif to save an animation)")


if __name__ == "__main__":
    main()
