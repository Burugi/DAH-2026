# -*- coding: utf-8 -*-
"""Single-agent centralized gym wrapper over the CybORG DroneSwarm worm sim.
obs   = compromised flags (n) + normalized step   -> shape (n+1,)
action= Discrete(n+1): 0 = sleep-all, k = clean drone (k-1) via RemoveOtherSessions (if owned)
Same action space for M1 heuristic / M2 PPO / M3 masked-PPO -> fair comparison.
"""
import sys
sys.path.insert(0, r"C:\workspace\DAH-2026")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from fleet import generate_fleet, starting_positions
from agents import RED_AGENTS, _action_index
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import DroneSwarmScenarioGenerator
from CybORG.Agents.Wrappers.PettingZooParallelWrapper import PettingZooParallelWrapper


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


class DroneCentralEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, seed=0):
        super().__init__()
        self.cfg = cfg
        self.base_seed = seed
        self.n = cfg["fleet"]["n_uav"] + cfg["fleet"]["n_ugv"]
        self.steps = cfg["steps"]
        self.action_space = spaces.Discrete(self.n + 1)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.n + 1,), dtype=np.float32)
        self._ep = 0
        self._owned = set()

    def _obs(self):
        self._owned = compromised_drones(self.cyborg, self.n)
        v = np.zeros(self.n + 1, dtype=np.float32)
        for d in self._owned:
            v[d] = 1.0
        v[self.n] = self.t / self.steps
        return v

    def reset(self, *, seed=None, options=None):
        s = self.base_seed + self._ep
        self._ep += 1
        fl, sim = self.cfg["fleet"], self.cfg["sim"]
        self.fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], self.steps, s, fl["grid"],
                                    self.cfg.get("attacks"), fl.get("max_link", 40))
        sg = DroneSwarmScenarioGenerator(
            num_drones=self.n, maximum_steps=self.steps,
            default_red_agent=RED_AGENTS[sim["red_class"]],
            starting_num_red=sim.get("starting_num_red", 1),
            red_spawn_rate=sim.get("red_spawn_rate", 0.05),
            max_length_data_links=sim.get("max_length_data_links", 40),
            starting_positions=starting_positions(self.fleet))
        self.cyborg = CybORG(sg, "sim", seed=int(s))
        self.env = PettingZooParallelWrapper(env=self.cyborg)
        self.env.reset()
        self.ip_to_drone = {ip: int(h.split("_")[-1]) for h, ip in self.cyborg.get_ip_map().items()
                            if h.startswith("drone_")}
        self.t = 0
        return self._obs(), {}

    def _actions_for(self, target):
        live = [a for a in self.env.active_agents if a in self.env.agent_actions]
        actions = {}
        for a in live:
            idx = _action_index(self.env, a)
            sleep_i = idx["Sleep"][0][0] if "Sleep" in idx else 0
            own = int(a.split("_")[-1])
            if target is not None and own == target and own in self._owned and "RemoveOtherSessions" in idx:
                actions[a] = idx["RemoveOtherSessions"][0][0]
            else:
                actions[a] = sleep_i
        return actions

    def step(self, action):
        target = None if int(action) == 0 else int(action) - 1
        actions = self._actions_for(target)
        _, rew, done, _ = self.env.step(actions)
        r = float(np.mean(list(rew.values()))) if rew else 0.0
        self.t += 1
        obs = self._obs()
        terminated = bool(done) and all(done.values())
        truncated = self.t >= self.steps
        return obs, r, terminated, truncated, {}

    def action_masks(self):
        m = np.zeros(self.n + 1, dtype=bool)
        m[0] = True
        for d in self._owned:
            m[d + 1] = True
        return m

    def heuristic_action(self):
        """M1: clean the lowest-index compromised drone (scripted, same action space)."""
        return (min(self._owned) + 1) if self._owned else 0


def eval_policy(cfg, pick_action, seeds, base_seed=1000):
    """Run a policy(callable env->action) over seeds; return mean compromised & reward."""
    comp, rew = [], []
    for s in seeds:
        env = DroneCentralEnv(cfg, seed=base_seed + s)
        obs, _ = env.reset()
        total = 0.0
        for _ in range(env.steps):
            a = pick_action(env, obs)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        comp.append(len(env._owned)); rew.append(total)
    return float(np.mean(comp)), float(np.mean(rew))
