# -*- coding: utf-8 -*-
"""C. Scalability: does the COORDINATOR advantage hold as swarm size grows? n in {12,24,36}.
Compare no-defense / uncoordinated(flat retake) / coordinator on simultaneous attack. Team metric."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions

base = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866


def make_red():
    AID = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}
    class R(brains._Red):
        VECS = ["W", "J", "B"]
        def get_action(self, obs, asp):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            o = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(AID[self.VECS[o % 3]])), obs)
    return R


def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node:
            return i
    c = idx.get("RetakeControl", [])
    return c[0][0] if c else sleep


def rollout(seed, cfg, defense):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red())
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); cf = []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}; acts = {}
        if defense == "coord":
            clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
            assign = {}; used = set()
            for c in comp:
                cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand:
                    assign[cand[0]] = c; used.add(cand[0])
            for a in live:
                i = int(a.split("_")[-1])
                if i in comp:
                    acts[a] = actions.make_blue_index(3, env, a, ctx)
                elif i in assign:
                    acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
                else:
                    acts[a] = actions.make_blue_index(1, env, a, ctx)
        elif defense == "uncoord":
            for a in live:
                i = int(a.split("_")[-1])
                acts[a] = actions.make_blue_index(3 if i in comp else (4 if comp else 1), env, a, ctx)
        else:
            for a in live:
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        if all(done.values()):
            break
    final = cf[-1]; auc = float(np.mean(cf)); av = max(0.0, 1 - final)
    return final, float(np.mean([1 - final, 1 - auc, COMP_F1])) * av


def ev(cfg, d):
    rs = [rollout(s, cfg, d) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(2))


SIZES = [(10, 5), (16, 8), (24, 12)]
print("=== scalability: coordinator advantage by swarm size (simultaneous, final compromise) ===")
print("n".ljust(5) + "nodef   uncoord   coord")
rows = []
for nu, ng in SIZES:
    cfg = dict(base); cfg["fleet"] = dict(base["fleet"]); cfg["fleet"]["n_uav"] = nu; cfg["fleet"]["n_ugv"] = ng
    nd = ev(cfg, "nodef")[0]; uc = ev(cfg, "uncoord")[0]; co = ev(cfg, "coord")[0]; n = nu + ng
    rows.append((n, nd, uc, co))
    print(f"{n:<5}{nd:.3f}   {uc:.3f}    {co:.3f}")
with open(os.path.join(OUT, "summary_scaling.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["n", "nodef", "uncoord", "coord"])
    for r in rows:
        wr.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)])
print("\nDONE_SCALING")
