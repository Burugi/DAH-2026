# -*- coding: utf-8 -*-
"""Actual REINFORCEMENT LEARNING (not imitation): train a per-drone GNN policy with REINFORCE
(reward = team 곱셈종합), many episodes. Logs learning curve + final perf vs BC-GNN/coordinator.
Also runs the agent loop many times for a confidence interval on the coordinator. CPU torch."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import torch, torch.nn as nn
import run
from agents import brains, actions
torch.manual_seed(1); np.random.seed(1)
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}
CLASS_AID = {0: 1, 1: 4}                 # 0 monitor, 1 retake(target)  (compromised auto self-clean)


def make_red(vectors):
    class R(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, asp):
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5, obs)
            o = int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[o % len(self.VECS)]])), obs)
    return R


def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1); return ((d < ml) & (d > 0)).astype(np.float32)


def features(comp, pos, ml, n, A):
    deg = A.sum(1); fg = len(comp) / n; cs = list(comp); X = np.zeros((n, 5), np.float32)
    for i in range(n):
        nbr = np.where(A[i])[0]; nc = sum(1 for j in nbr if j in comp)
        X[i, 0] = 1.0 if i in comp else 0.0; X[i, 1] = nc / max(1, deg[i]); X[i, 2] = fg
        X[i, 3] = 1.0 if (i not in comp and nc > 0) else 0.0
        X[i, 4] = (min(np.linalg.norm(pos[i] - pos[c]) for c in cs) / ml) if (cs and i not in comp) else 1.0
    return X


def retake_nearest(env, a, i, comp, pos, ip2d, sleep):
    if not comp: return sleep
    tgt = min(comp, key=lambda c: np.linalg.norm(pos[i] - pos[c]))
    idx = actions.action_index_map(env, a)
    for j, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == tgt: return j
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep


class GNN(nn.Module):
    def __init__(s):
        super().__init__(); s.l1 = nn.Linear(5, 48); s.m1 = nn.Linear(5, 48)
        s.l2 = nn.Linear(48, 48); s.m2 = nn.Linear(48, 48); s.out = nn.Linear(48, 2)
    def forward(s, X, A):
        deg = A.sum(1, keepdim=True).clamp(min=1)
        h = torch.relu(s.l1(X) + s.m1(A @ X / deg)); h = torch.relu(s.l2(h) + s.m2(A @ h / deg))
        return s.out(h)


def episode(model, seed, vec, train=True):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red(vec)); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    cf, af = [], []; logps = []; k = len(vec)
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        X = torch.tensor(features(comp, pos, ml, n, A)); At = torch.tensor(A)
        logit = model(X, At); prob = torch.softmax(logit, 1)
        red_jam = sum(1 for i in comp if vec[i % k] in JAM_VECS)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if train:
                m = torch.distributions.Categorical(prob[i]); c = int(m.sample()); logps.append(m.log_prob(torch.tensor(c)))
            else:
                c = int(prob[i].argmax())
            acts[a] = retake_nearest(env, a, i, comp, pos, ip2d, sleep) if c == 1 else actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - red_jam) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    ret = float(np.mean([1 - final, 1 - auc, COMP_F1])) * av
    return ret, logps


TRAIN_ATK = [["W"], ["J"], ["W", "J", "B"]]
print("=== REINFORCE training (reward = 곱셈종합) ==="); model = GNN()
opt = torch.optim.Adam(model.parameters(), lr=3e-3); base = 0.5; curve = []
rng = np.random.default_rng(2)
for ep in range(400):
    vec = TRAIN_ATK[int(rng.integers(len(TRAIN_ATK)))]; seed = 3010 + int(rng.integers(40))
    ret, logps = episode(model, seed, vec, train=True)
    if logps:
        adv = ret - base; loss = -adv * torch.stack(logps).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    base = 0.95 * base + 0.05 * ret
    if (ep + 1) % 40 == 0:
        curve.append((ep + 1, base)); print(f"  ep {ep+1:3d}  baseline 곱셈 {base:.3f}")

EVAL_ATK = [["W"], ["J"], ["W", "J", "B"]]; EVAL_SEEDS = [3000, 3001, 3002, 3003, 3004]
rl_final = float(np.mean([episode(model, s, v, train=False)[0] for s in EVAL_SEEDS for v in EVAL_ATK]))
print(f"\nRL(REINFORCE GNN) 최종 곱셈종합: {rl_final:.3f}")
print("  (참조) BC-GNN 0.691 · 적응 0.68 · 코디네이터(전문가) ~0.83")

# ---- confidence interval on coordinator: run agent loop many times ----
def coord_once(seed, vec):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red(vec)); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    cf, af = [], []; k = len(vec)
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; red_jam = sum(1 for i in comp if vec[i % k] in JAM_VECS)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
        assign = {}; used = set()
        for c in comp:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx)
            elif i in assign: acts[a] = retake_nearest(env, a, i, comp, pos, ip2d, sleep)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - red_jam) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

samples = [coord_once(s, v) for s in range(3000, 3030) for v in EVAL_ATK]
m = float(np.mean(samples)); sd = float(np.std(samples)); ci = 1.96 * sd / np.sqrt(len(samples))
print(f"\n코디네이터 {len(samples)}회 반복: 곱셈 {m:.3f} ± {ci:.3f} (95% CI), std {sd:.3f}")
with open(os.path.join(OUT, "summary_rl.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["ep", "baseline_곱셈"]);
    for e, b in curve: wr.writerow([e, round(b, 3)])
    wr.writerow([]); wr.writerow(["RL_final", round(rl_final, 3)])
    wr.writerow(["coordinator_mean", round(m, 3)]); wr.writerow(["coordinator_CI95", round(ci, 3)])
print("DONE_RL")
