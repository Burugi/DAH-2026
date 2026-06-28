"""Action gallery: force each red/blue action in isolation, save a GIF and measure
its effect, to verify every action actually runs and that the simulations differ.

    python src/gallery.py

Writes results/gallery/<team>_<id>_<name>/ (log.csv, arrays.npz, meta.json, figs/
animation.gif) for every action, plus results/gallery/action_info.csv summarising
each action's sim primitive, measured effect and whether it differs from the
Sleep/passive baseline.

Note (two-channel design): exploit-family red actions and blue retake/remove are
visible in the GIF as the fill colour changing (compromise spread / recovery). Red
Jam/Block act on the CybORG network channel (FloodBandwidth/BlockTraffic) and show
up in the metrics rather than as a coloured ring, because the purple jam ring comes
from the synthetic telemetry overlay, which is turned off here so the gallery shows
only agent-driven effects.
"""
import os, sys, csv
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from CybORG.Agents import BaseAgent
import run
from viz import render
from agents import actions
from agents.brains import RED_BRAINS

GAL = os.path.join(RESULTS, "gallery")

# small fleet + no synthetic attacks -> the GIF shows only agent-driven effects
GAL_CFG = {
    "name": "gallery",
    "fleet": {"n_uav": 6, "n_ugv": 2, "grid": 60, "max_link": 40},
    "steps": 30,
    "seeds": [0],
    "sim": {"starting_num_red": 1, "red_spawn_rate": 0.1, "max_length_data_links": 40},
    "defense": {"detector": "multisensor", "snr_thresh": 6, "gps_thresh": 8, "response": "safe_mode"},
    "attacks": [],
}

RED_PRIM = {0: "Sleep", 1: "Sleep(observe)", 2: "ExploitDroneVulnerability",
            3: "ExploitDroneVulnerability", 4: "ExploitDroneVulnerability",
            5: "SeizeControl", 6: "ExploitDroneVulnerability", 7: "FloodBandwidth",
            8: "FloodBandwidth", 9: "BlockTraffic", 10: "SeizeControl",
            11: "ExploitDroneVulnerability", 12: "ExploitDroneVulnerability",
            13: "FloodBandwidth"}
BLUE_PRIM = {0: "Sleep", 1: "Sleep(observe)", 2: "Sleep(observe)",
             3: "RemoveOtherSessions", 4: "RetakeControl", 5: "RetakeControl",
             6: "BlockTraffic", 7: "AllowTraffic", 8: "Sleep(decoy)"}

_FORCED = {"k": 0}


class ForcedRed(BaseAgent):
    """Red agent that always performs action _FORCED['k'] (with the minimal
    prerequisite cycle for Seize/Persist, which need a prior exploit)."""
    def __init__(self, name, np_random=None):
        super().__init__(name, np_random)
        self.mem, self.t = {}, 0

    def get_action(self, obs, action_space):
        k = _FORCED["k"]
        seq = {5: [2, 5], 10: [2, 5, 10]}.get(k, [k])
        aid = seq[self.t % len(seq)]
        act, _ = actions.make_red_action(aid, obs, self.name, self.mem, self.np_random)
        self.t += 1
        return act

    def train(self, results): pass
    def end_episode(self): self.mem, self.t = {}, 0
    def set_initial_values(self, action_space, observation): pass


def forced_rollout(cfg, seed, red_class, blue_k):
    """One episode with a fixed red class and a fixed blue action id (None=passive)."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red_class)
    n = fleet["n"]
    reward = np.zeros(cfg["steps"])
    red_owned = np.zeros((cfg["steps"], n), np.int8)
    for t in range(cfg["steps"]):
        ctx = {"compromised": run.compromised_drones(cyborg, n), "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {} if blue_k is None else {a: actions.make_blue_index(blue_k, env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        reward[t] = float(np.mean(list(rew.values()))) if rew else 0.0
        for d in run.compromised_drones(cyborg, n):
            red_owned[t, d] = 1
        if all(done.values()):
            reward, red_owned = reward[:t + 1], red_owned[:t + 1]
            run.truncate(fleet, t + 1)
            break
    df = run.run_defense(cfg, fleet, seed, red_owned)
    return fleet, reward, red_owned, df


def _one(out, red_class, blue_k, red_label, blue_label):
    res = {0: forced_rollout(GAL_CFG, 0, red_class, blue_k)}
    m = run.save_run(GAL_CFG, out, red_label, blue_label, res)
    render.save_gif(out)
    return m["final_compromise"], m["blue_reward_total"]


def main():
    os.makedirs(GAL, exist_ok=True)
    rows = []

    # red gallery: focal red action vs passive blue, compared to red Sleep baseline
    _FORCED["k"] = 0
    base_fc, base_r = _one(os.path.join(GAL, "red_00_Sleep"), ForcedRed, None, "forced:Sleep", "passive")
    rows.append(["red", 0, "Sleep", RED_PRIM[0], round(base_fc, 3), round(base_r, 1), "baseline"])
    for k in range(1, actions.RED_N):
        name = actions.RED_CATALOG[k][0]
        _FORCED["k"] = k
        fc, r = _one(os.path.join(GAL, f"red_{k:02d}_{name}"), ForcedRed, None, f"forced:{name}", "passive")
        distinct = "yes" if (abs(fc - base_fc) > 0.01 or abs(r - base_r) > 1.0) else "no(=Sleep)"
        rows.append(["red", k, name, RED_PRIM[k], round(fc, 3), round(r, 1), distinct])
        print(f"  red  {k:2} {name:14} fc={fc:.2f} blueR={r:.0f} {distinct}")

    # blue gallery: focal blue action vs an active rule-red, compared to passive blue
    rule_red = RED_BRAINS["rule"]
    bbase_fc, bbase_r = _one(os.path.join(GAL, "blue_00_Passive"), rule_red, None, "rule", "passive")
    rows.append(["blue", -1, "Passive", "Sleep", round(bbase_fc, 3), round(bbase_r, 1), "baseline"])
    for k in range(actions.BLUE_DECISION_N):
        name = actions.BLUE_CATALOG[k][0]
        fc, r = _one(os.path.join(GAL, f"blue_{k:02d}_{name}"), rule_red, k, "rule", f"forced:{name}")
        distinct = "yes" if (abs(fc - bbase_fc) > 0.01 or abs(r - bbase_r) > 1.0) else "no(=Passive)"
        rows.append(["blue", k, name, BLUE_PRIM[k], round(fc, 3), round(r, 1), distinct])
        print(f"  blue {k:2} {name:16} fc={fc:.2f} blueR={r:.0f} {distinct}")

    cols = ["team", "id", "name", "sim_primitive", "final_compromise", "blue_reward", "distinct_vs_baseline"]
    with open(os.path.join(GAL, "action_info.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([cols] + rows)
    print(f"\n-> {os.path.relpath(GAL, ROOT)}/  ({len(rows)} actions, GIF each + action_info.csv)")


if __name__ == "__main__":
    main()
