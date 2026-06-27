"""Lightweight tabular Q-learning for the 'rl' agent type (pure numpy, local).

This is the "강화학습" third agent: a tiny Monte-Carlo control Q-table over a
small discrete state. It trains in a few hundred short episodes on CPU, then is
cached and evaluated greedily inside the 3x3 sweep.

  python rl.py train [config]   ->  results/rl_blue.npz, results/rl_red.npz

Training is self-play-free and kept simple: RL-blue learns against rule-red, and
RL-red learns against rule-blue. The learned policies are then frozen for eval.
"""
import os, sys, argparse
import numpy as np
import yaml

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

BLUE_Q = os.path.join(ROOT, "results", "rl_blue.npz")
RED_Q = os.path.join(ROOT, "results", "rl_red.npz")
DEFAULT_CFG = os.path.join(SRC, "configs", "sweep.yaml")


class QTable:
    def __init__(self, n_actions, alpha=0.3, gamma=0.95):
        self.n = int(n_actions)
        self.alpha, self.gamma = alpha, gamma
        self.q = {}                                  # state tuple -> values[n]

    def row(self, s):
        if s not in self.q:
            self.q[s] = np.zeros(self.n)
        return self.q[s]

    def act(self, s, eps, rng):
        if eps and rng.random() < eps:
            return int(rng.integers(self.n))
        return int(np.argmax(self.row(s)))

    def learn(self, traj):
        """Monte-Carlo update over one trajectory of (state, action, reward)."""
        g = 0.0
        for s, a, r in reversed(traj):
            g = r + self.gamma * g
            row = self.row(s)
            row[a] += self.alpha * (g - row[a])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = list(self.q.keys())
        vals = np.array([self.q[k] for k in keys]) if keys else np.zeros((0, self.n))
        np.savez(path, keys=np.array(keys, dtype=object), vals=vals,
                 n=self.n, alpha=self.alpha, gamma=self.gamma)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        q = cls(int(z["n"]), float(z["alpha"]), float(z["gamma"]))
        for k, v in zip(z["keys"], z["vals"]):
            q.q[tuple(int(x) for x in k)] = np.array(v, dtype=float)
        return q


# --------------------------------------------------------- training ---
def _train_blue_episode(cfg, seed, q, eps, rng):
    """rule-red vs RL-blue; MC-update q from the shared blue reward."""
    import run
    from agents import brains, actions
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RuleRed)
    n = fleet["n"]
    trajs = {}
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts, picks = {}, {}
        for a in live:
            s = brains.blue_state(a, ctx)
            aid = q.act(s, eps, rng)
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
            picks[a] = (s, aid)
        _, rew, done, _ = env.step(acts)
        r = float(np.mean(list(rew.values()))) if rew else 0.0
        for a, (s, aid) in picks.items():
            trajs.setdefault(a, []).append((s, aid, r))
        if all(done.values()):
            break
    for tr in trajs.values():
        q.learn(tr)


def _train_red_episode(cfg, seed, q, eps, rng):
    """RL-red vs rule-blue; reward = gain in compromised drones."""
    import run
    from agents import brains, actions
    brains.set_red_q(q, eps, rng, train=True)
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RLRed)
    n = fleet["n"]
    trajs = {}
    prev = len(run.compromised_drones(cyborg, n))
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(brains.blue_decide("rule", env, a, ctx),
                                           env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)             # red get_action runs here -> fills stash
        stash = brains.pop_red_stash()               # (name, state, aid) taken this step
        now = len(run.compromised_drones(cyborg, n))
        r = float(now - prev)                        # +1 per newly owned drone this step
        prev = now
        for name, s, aid in stash:
            trajs.setdefault(name, []).append((s, aid, r))
        if all(done.values()):
            break
    brains.set_red_q(None, 0.0, None, train=False)
    for tr in trajs.values():
        q.learn(tr)


def train_rl(cfg, episodes=200, verbose=True):
    from agents import actions
    q_blue, q_red = QTable(actions.BLUE_DECISION_N), QTable(actions.RED_N)
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        _train_blue_episode(cfg, ep, q_blue, eps, rng)
        _train_red_episode(cfg, ep, q_red, eps, rng)
        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            print(f"  rl train {ep + 1}/{episodes}  "
                  f"|states| blue={len(q_blue.q)} red={len(q_red.q)}")
    q_blue.save(BLUE_Q)
    q_red.save(RED_Q)
    print(f"-> {BLUE_Q}\n-> {RED_Q}")
    return q_blue, q_red


def ensure_trained(cfg, episodes=200, fresh=False):
    """Train if a cache is missing (or fresh=True); otherwise load both tables."""
    if not fresh and os.path.isfile(BLUE_Q) and os.path.isfile(RED_Q):
        return QTable.load(BLUE_Q), QTable.load(RED_Q)
    print("rl: training policies...")
    return train_rl(cfg, episodes)


def main():
    ap = argparse.ArgumentParser(description="Train and cache the tabular-Q rl policies.")
    ap.add_argument("cmd", nargs="?", default="train", choices=["train"])
    ap.add_argument("config", nargs="?", default=DEFAULT_CFG)
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()
    train_rl(yaml.safe_load(open(a.config, encoding="utf-8")), a.episodes)


if __name__ == "__main__":
    main()
