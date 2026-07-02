# -*- coding: utf-8 -*-
"""#7 (real) Graph Neural Network defender.
A message-passing GNN over the drone comms graph (nodes=drones, edges=links within range).
Heuristic floor: own-compromised -> RemoveSessions(3). Else: the GNN aggregates NEIGHBOUR
compromise via the graph and picks a proactive action (retake/block/decoy) per node.
Trained with REINFORCE on n=18, then evaluated on swarm sizes 12/18/30 to test size invariance
(the GNN's selling point) against the tabular hybrid (which overfit to size 18) and the heuristic.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, copy, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.distributions import Categorical
import run
from agents import brains, actions
from agents.rl import QTable

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
PRO = [4, 6, 8]               # retake / block / decoy
torch.manual_seed(0); np.random.seed(0)


def adjacency(pos, max_link):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    A = (d < max_link).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = A / A.sum(1, keepdims=True)
    return torch.tensor(A, dtype=torch.float32)


def node_feats(comp, n):
    x = np.zeros((n, 2), np.float32)
    for d in comp:
        if d < n: x[d, 0] = 1.0
    x[:, 1] = len(comp) / max(1, n)
    return torch.tensor(x, dtype=torch.float32)


class GNNPolicy(nn.Module):
    def __init__(self, fdim=2, hid=16, nact=len(PRO)):
        super().__init__()
        self.w1 = nn.Linear(fdim, hid); self.w2 = nn.Linear(hid, hid); self.head = nn.Linear(hid, nact)
    def forward(self, x, A):
        h = torch.relu(self.w1(A @ x))
        h = torch.relu(self.w2(A @ h))
        return self.head(h)                       # (n, nact) logits


def gnn_rollout(cfg, seed, policy, train=False):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, brains.RuleRed)
    n = fleet["n"]; max_link = cfg["fleet"].get("max_link", 40)
    logps, total = [], 0.0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        A = adjacency(pos, max_link); x = node_feats(comp, n)
        logits = policy(x, A); dist = Categorical(logits=logits)
        samp = dist.sample() if train else torch.argmax(logits, dim=1)
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        for a in live:
            own = int(a.split("_")[-1])
            if own in comp:
                aid = 3
            else:
                aid = PRO[int(samp[own].item())]
                if train: logps.append(dist.log_prob(samp)[own])
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        total += float(np.mean(list(rew.values()))) if rew else 0.0
        if all(done.values()):
            break
    comp_frac = len(run.compromised_drones(cyborg, n)) / n
    return total, comp_frac, logps


def train_gnn(cfg, episodes=180):
    policy = GNNPolicy(); opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    baseline = None
    for ep in range(episodes):
        total, _, logps = gnn_rollout(cfg, ep, policy, train=True)
        if not logps:
            continue
        baseline = total if baseline is None else 0.95 * baseline + 0.05 * total
        adv = total - baseline
        loss = -adv * torch.stack(logps).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 30 == 0:
            print(f"  gnn train {ep+1}/{episodes}  return~{baseline:.0f}")
    return policy


# tabular hybrid (for fair size comparison)
def train_hybrid(cfg, episodes=300):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed); n = fleet["n"]; trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}
            for a in live:
                own = int(a.split("_")[-1])
                if own in comp: aid = 3
                else:
                    s = brains.blue_state(a, ctx); idx = q.act(s, eps, rng); aid = PRO[idx]; picks[a] = (s, idx)
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, float(np.mean(list(rew.values()))) if rew else 0.0))
            if all(done.values()): break
        for tr in trajs.values(): q.learn(tr)
    return q


def eval_perAgent(cfg, blue_fn, seeds):
    cs = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, brains.RuleRed); n = fleet["n"]
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            live = [a for a in env.active_agents if a in env.agent_actions]
            acts = {a: actions.make_blue_index(blue_fn(env, a, ctx), env, a, ctx) for a in live}
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


def eval_gnn(cfg, policy, seeds):
    return float(np.mean([gnn_rollout(cfg, s, policy, train=False)[1] for s in seeds]))


print("Training GNN-hybrid (REINFORCE, n=18) ...")
policy = train_gnn(BASE)
print("Training tabular hybrid (n=18) ...")
q_hyb = train_hybrid(BASE)

def f_heur(env, a, ctx): return brains.blue_decide("rule", env, a, ctx)
def f_tab(env, a, ctx):
    own = int(a.split("_")[-1])
    return 3 if own in ctx["compromised"] else PRO[q_hyb.act(brains.blue_state(a, ctx), 0.0, None)]

sizes = [(8, 4), (12, 6), (20, 10)]            # 12 / 18 / 30
labels_n = [nu + ng for nu, ng in sizes]
rows = {"heuristic": [], "tabular-hybrid(18)": [], "GNN-hybrid(18)": []}
print(f"\n=== size generalization (final compromise) ===")
print(f"{'defender':22}" + "".join(f"{n:>8}" for n in labels_n))
for nu, ng in sizes:
    cfg = copy.deepcopy(BASE); cfg["fleet"]["n_uav"] = nu; cfg["fleet"]["n_ugv"] = ng; cfg["attacks"] = []
    rows["heuristic"].append(eval_perAgent(cfg, f_heur, EVAL))
    rows["tabular-hybrid(18)"].append(eval_perAgent(cfg, f_tab, EVAL))
    rows["GNN-hybrid(18)"].append(eval_gnn(cfg, policy, EVAL))
for k, v in rows.items():
    print(f"{k:22}" + "".join(f"{x:8.3f}" for x in v))

plt.figure(figsize=(7.4, 4.5))
for k, col, mk in [("heuristic", "seagreen", "o"), ("tabular-hybrid(18)", "royalblue", "s"), ("GNN-hybrid(18)", "purple", "^")]:
    plt.plot(labels_n, rows[k], mk + "-", color=col, lw=1.9, label=k)
plt.xlabel("swarm size (drones)"); plt.ylabel("final compromised fraction (lower=better)")
plt.title("#7 GNN size generalization: trained on 18, evaluated on 12/18/30")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig16_gnn_size.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_gnn.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["defender"] + [str(n) for n in labels_n])
    for k, v in rows.items(): w.writerow([k] + [round(x, 3) for x in v])
print("\nSaved fig16_gnn_size.png, summary_gnn.csv")
print("RESULT:", {k: [round(x, 3) for x in v] for k, v in rows.items()})
