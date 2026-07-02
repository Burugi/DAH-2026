# -*- coding: utf-8 -*-
"""Deep RL roadmap test: SB3 PPO (pure) vs PPO-hybrid (heuristic floor + deep policy).
Single-agent centralized gym wrappers over the code-team harness. Compare to the
tabular results (heuristic / RL-multi / tabular-hybrid).
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import gymnasium as gym
from gymnasium import spaces
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
N = cfg["fleet"]["n_uav"] + cfg["fleet"]["n_ugv"]
STEPS = cfg["steps"]
PROACTIVE = [4, 6, 8]   # retake / block / decoy
TRAIN_STEPS = 40000


class Base(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base=0):
        super().__init__()
        self.base = base; self._ep = 0
        self.observation_space = spaces.Box(0.0, 1.0, (N + 1,), np.float32)

    def reset(self, *, seed=None, options=None):
        s = self.base + self._ep; self._ep += 1
        self.fleet, self.cyborg, self.env, self.ip2d = run.build_env(cfg, s, brains.RuleRed)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        self._comp = run.compromised_drones(self.cyborg, N)
        v = np.zeros(N + 1, np.float32)
        for d in self._comp:
            v[d] = 1.0
        v[N] = self.t / STEPS
        return v

    def _ctx(self):
        return {"compromised": self._comp, "ip_to_drone": self.ip2d, "n": N}

    def _live(self):
        return [a for a in self.env.active_agents if a in self.env.agent_actions]

    def _apply(self, acts):
        _, rew, done, _ = self.env.step(acts)
        self.t += 1
        r = float(np.mean(list(rew.values()))) if rew else 0.0
        return r, (bool(done) and all(done.values())), self.t >= STEPS


class PurePPOEnv(Base):
    """Pure deep RL: action 0 = monitor all, k = RemoveSessions on drone k-1."""
    def __init__(self, base=0):
        super().__init__(base)
        self.action_space = spaces.Discrete(N + 1)

    def step(self, action):
        ctx = self._ctx(); target = None if int(action) == 0 else int(action) - 1
        acts = {}
        for a in self._live():
            own = int(a.split("_")[-1])
            aid = 3 if (target is not None and own == target) else 1
            acts[a] = actions.make_blue_index(aid, self.env, a, ctx)
        r, term, trunc = self._apply(acts)
        return self._obs(), r, term, trunc, {}


class HybridPPOEnv(Base):
    """Deep hybrid: own-compromised -> Remove(3) (heuristic core); else PPO picks one PROACTIVE."""
    def __init__(self, base=0):
        super().__init__(base)
        self.action_space = spaces.Discrete(len(PROACTIVE))

    def step(self, action):
        ctx = self._ctx(); aid_else = PROACTIVE[int(action)]
        acts = {}
        for a in self._live():
            own = int(a.split("_")[-1])
            aid = 3 if own in self._comp else aid_else
            acts[a] = actions.make_blue_index(aid, self.env, a, ctx)
        r, term, trunc = self._apply(acts)
        return self._obs(), r, term, trunc, {}


def evaluate(model, EnvCls, seeds):
    comps, rews = [], []
    for s in seeds:
        e = EnvCls(base=s); obs, _ = e.reset(); tot = 0.0
        for _ in range(STEPS):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = e.step(int(a)); tot += r
            if term or trunc:
                break
        comps.append(len(e._comp) / N); rews.append(tot)
    return float(np.mean(comps)), float(np.std(comps)), float(np.mean(rews))


res = {}
for name, EnvCls in [("PPO-pure\n(deep, central)", PurePPOEnv), ("PPO-hybrid\n(heuristic+deep)", HybridPPOEnv)]:
    print("Training", name.replace("\n", " "), "...")
    env = EnvCls(base=0)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=128, gae_lambda=0.95,
                gamma=0.95, ent_coef=0.01, learning_rate=3e-4, verbose=0, seed=0)
    model.learn(total_timesteps=TRAIN_STEPS)
    c, cs, r = evaluate(model, EnvCls, EVAL)
    res[name] = (c, cs, r)
    print(f"  {name.replace(chr(10),' ')}: final_comp={c:.3f}  reward={r:.0f}")

# reference numbers from prior tabular runs
REF = [("heuristic\n(multi)", 0.356, "seagreen"),
       ("RL-multi\n(tabular)", 0.867, "indianred"),
       ("tabular-hybrid", 0.311, "royalblue")]

labels = [r[0] for r in REF] + list(res.keys())
comps = [r[1] for r in REF] + [res[k][0] for k in res]
cstd = [0, 0, 0] + [res[k][1] for k in res]
cols = [r[2] for r in REF] + ["purple", "teal"]

plt.figure(figsize=(9, 4.8))
x = np.arange(len(labels))
plt.bar(x, comps, yerr=cstd, color=cols, alpha=0.85, capsize=4)
plt.xticks(x, labels, fontsize=8)
plt.ylabel("final compromised fraction (lower=better)")
for i, v in enumerate(comps): plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
plt.title("Roadmap test: deep PPO vs tabular vs heuristic (defender, vs rule-red, held-out)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig11_ppo_roadmap.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_ppo.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["defender", "final_compromise", "std", "reward"])
    for k in res: w.writerow([k.replace("\n", " "), round(res[k][0], 3), round(res[k][1], 3), round(res[k][2], 1)])
print("\nSaved fig11_ppo_roadmap.png, summary_ppo.csv")
print("RESULT:", {k.replace(chr(10), ' '): (round(v[0], 3), round(v[2], 0)) for k, v in res.items()})
