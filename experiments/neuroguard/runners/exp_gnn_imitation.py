# -*- coding: utf-8 -*-
"""Limit-breaking: GNN with IMITATION warmup of HYBRID+.
HYBRID+ (richer tabular) is strong at the trained size but overfits small swarms.
A GNN is node-wise (size-invariant by construction). We train a GNN to IMITATE HYBRID+
(supervised on the comms graph), then evaluate across sizes 12/18/30 to test whether the
GNN keeps HYBRID+'s strength while generalizing across cluster size.
"""
import sys, os, copy, random
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import run
from agents import brains, actions
from agents.rl import QTable
torch.manual_seed(0); np.random.seed(0); random.seed(0)

OUT = r"C:\workspace\DAH2026_exp"
BASE = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [1000, 1001, 1002, 1003, 1004]
PRO = [4, 6, 8]; MU = 0.3


def adj(pos, ml):
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    A = (d < ml).astype(np.float32); np.fill_diagonal(A, 1.0); A /= A.sum(1, keepdims=True)
    return torch.tensor(A)

def feats(comp, n, frac):
    x = np.zeros((n, 2), np.float32)
    for d in comp:
        if d < n: x[d, 0] = 1.0
    x[:, 1] = frac
    return torch.tensor(x)

def inf_nbr(i, comp, pos, ml):
    return sum(1 for d in comp if d != i and np.linalg.norm(pos[i] - pos[d]) < ml)

def rstate(i, comp, pos, n, ml):
    return (1 if i in comp else 0, min(3, inf_nbr(i, comp, pos, ml)), min(3, int(len(comp) / n * 4)))


# ---- 1) train HYBRID+ (richer tabular) to be the imitation teacher ----
def train_hybrid_plus(cfg, episodes=260):
    q = QTable(len(PRO)); rng = np.random.default_rng(0)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.8 * episodes))
        fleet, cyborg, env, ip2d = run.build_env(cfg, ep, brains.RuleRed)
        n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); trajs = {}
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            live = [a for a in env.active_agents if a in env.agent_actions]; acts, picks = {}, {}; blk = 0
            for a in live:
                i = int(a.split("_")[-1])
                if i in comp: aid = 3
                else:
                    idx = q.act(rstate(i, comp, pos, n, ml), eps, rng); aid = PRO[idx]; picks[a] = (rstate(i, comp, pos, n, ml), idx)
                if aid in (6, 8): blk += 1
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            r = (float(np.mean(list(rew.values()))) if rew else 0.0) - MU * blk
            for a, (s, idx) in picks.items(): trajs.setdefault(a, []).append((s, idx, r))
            if all(done.values()): break
        for tr in trajs.values(): q.learn(tr)
    return q


# ---- 2) GNN ----
class GNN(nn.Module):
    def __init__(self, fdim=2, hid=24, nact=3):
        super().__init__(); self.w1 = nn.Linear(fdim, hid); self.w2 = nn.Linear(hid, hid); self.head = nn.Linear(hid, nact)
    def forward(self, x, A):
        h = torch.relu(self.w1(A @ x)); h = torch.relu(self.w2(A @ h)); return self.head(h)


def collect(cfg, q_teacher, episodes=80):
    data = []
    for ep in range(episodes):
        fleet, cyborg, env, ip2d = run.build_env(cfg, 5000 + ep, brains.RuleRed)
        n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            A = adj(pos, ml); X = feats(comp, n, min(3, int(len(comp) / n * 4)) / 3.0)
            y = -np.ones(n, dtype=np.int64); live = [a for a in env.active_agents if a in env.agent_actions]; acts = {}
            for a in live:
                i = int(a.split("_")[-1])
                if i in comp: aid = 3
                else:
                    idx = q_teacher.act(rstate(i, comp, pos, n, ml), 0.0, None); aid = PRO[idx]; y[i] = idx
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            data.append((X, A, torch.tensor(y)))
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
    return data


def train_gnn(data, epochs=40):
    gnn = GNN(); opt = torch.optim.Adam(gnn.parameters(), lr=3e-3); ce = nn.CrossEntropyLoss()
    for e in range(epochs):
        random.shuffle(data); tot = 0.0
        for X, A, y in data:
            m = y >= 0
            if m.sum() == 0: continue
            logits = gnn(X, A); loss = ce(logits[m], y[m])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (e + 1) % 10 == 0: print(f"  gnn imitate epoch {e+1}/{epochs} loss~{tot/max(1,len(data)):.3f}")
    return gnn


def eval_decide(cfg, decide, seeds, gnn=None):
    cs = []
    for s in seeds:
        fleet, cyborg, env, ip2d = run.build_env(cfg, s, brains.RuleRed); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
        for t in range(cfg["steps"]):
            comp = run.compromised_drones(cyborg, n); ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
            pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
            live = [a for a in env.active_agents if a in env.agent_actions]
            if gnn is not None:
                with torch.no_grad():
                    samp = torch.argmax(gnn(feats(comp, n, min(3, int(len(comp)/n*4))/3.0), adj(pos, ml)), 1)
            acts = {}
            for a in live:
                i = int(a.split("_")[-1])
                if gnn is not None:
                    aid = 3 if i in comp else PRO[int(samp[i])]
                else:
                    aid = decide(i, comp, pos, n, ml)
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
            _, rew, done, _ = env.step(acts)
            if all(done.values()): break
        cs.append(len(run.compromised_drones(cyborg, n)) / n)
    return float(np.mean(cs))


print("1) train HYBRID+ teacher ..."); q_hp = train_hybrid_plus(BASE)
print("2) collect imitation data ..."); data = collect(BASE, q_hp)
print(f"   {len(data)} graph snapshots"); print("3) train GNN (imitate HYBRID+) ..."); gnn = train_gnn(data)

def d_heur(i, comp, pos, n, ml): return brains.blue_decide("rule", None, f"x_{i}", {"compromised": comp, "n": n}) if False else (3 if i in comp else (4 if comp else 1))
def d_hp(i, comp, pos, n, ml): return 3 if i in comp else PRO[q_hp.act(rstate(i, comp, pos, n, ml), 0.0, None)]

sizes = [(8, 4), (12, 6), (20, 10)]; labs = [nu+ng for nu, ng in sizes]
res = {"휴리스틱": [], "HYBRID+ (tabular)": [], "GNN-모방 (크기불변)": []}
print("\n=== size generalization (vs rule-red) ===")
print("size   휴리스틱   HYBRID+   GNN-모방")
for nu, ng in sizes:
    cfg = copy.deepcopy(BASE); cfg["fleet"]["n_uav"] = nu; cfg["fleet"]["n_ugv"] = ng; cfg["attacks"] = []
    h = eval_decide(cfg, d_heur, EVAL); hp = eval_decide(cfg, d_hp, EVAL); g = eval_decide(cfg, None, EVAL, gnn=gnn)
    res["휴리스틱"].append(h); res["HYBRID+ (tabular)"].append(hp); res["GNN-모방 (크기불변)"].append(g)
    print(f"{nu+ng:>4}    {h:.3f}    {hp:.3f}    {g:.3f}")

plt.figure(figsize=(7.4, 4.6))
for k, col, mk in [("휴리스틱", "seagreen", "o"), ("HYBRID+ (tabular)", "purple", "s"), ("GNN-모방 (크기불변)", "crimson", "^")]:
    plt.plot(labs, res[k], mk + "-", color=col, lw=2, label=k)
plt.xlabel("군집 크기"); plt.ylabel("최종 점령 (낮을수록 좋음)")
plt.title("한계돌파: GNN 모방 워밍업이 크기 과적합을 해결하나")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig25_gnn_imitation.png"), dpi=130); plt.close()
import csv
with open(os.path.join(OUT, "summary_gnn_imitation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["size"] + list(res.keys()))
    for i, s in enumerate(labs): w.writerow([s] + [round(res[k][i], 3) for k in res])
print("\nSaved fig25_gnn_imitation.png, summary_gnn_imitation.csv")
print("RESULT:", {k: [round(v, 3) for v in vs] for k, vs in res.items()})
