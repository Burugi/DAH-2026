# -*- coding: utf-8 -*-
"""A. Can we RAISE the connectivity ceiling? Against fragmentation (K hubs jammed), dedicate R drones
as RELAYS (reconnect isolated nodes) but relay drones can't defend (tradeoff). Sweep R. Team metric."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions

cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866; FRAG_K = 4


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


def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return (d < ml) & (d > 0)


def largest_comp(present, A, n):
    seen = set(); best = set()
    for s in present:
        if s in seen:
            continue
        comp = set(); st = [s]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u); comp.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen:
                    st.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node:
            return i
    c = idx.get("RetakeControl", [])
    return c[0][0] if c else sleep


def rollout(seed, R_relay):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, make_red())
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:FRAG_K])
    cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs
        big = largest_comp(present, A, n) if present else set()
        isolated = present - big
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean_all = [int(a.split("_")[-1]) for a in live
                     if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in hubs and int(a.split("_")[-1]) not in isolated]
        relays = set()
        for x in list(isolated)[:R_relay]:
            cand = [d for d in clean_all if d not in relays]
            if not cand:
                break
            r = min(cand, key=lambda d: np.linalg.norm(pos[d] - pos[x]))
            relays.add(r); isolated.discard(x)
        defenders = [d for d in clean_all if d not in relays]
        unreachable = hubs | isolated
        reachable_comp = comp - unreachable
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        assign = {}; used = set()
        for c in reachable_comp:
            cand = sorted([d for d in defenders if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand:
                assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable:
                acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in relays:
                acts[a] = actions.make_blue_index(1, env, a, ctx); continue
            if i in comp:
                acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - len(unreachable)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()):
            break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return final, av, float(np.mean([1 - final, 1 - auc, COMP_F1])) * av


def ev(R):
    rs = [rollout(s, R) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))


print(f"=== connectivity-maintenance tradeoff (frag K={FRAG_K}, sweep relay budget R) ===")
print("R_relay".ljust(9) + "compromise  avail   곱셈")
rows = []
for R in [0, 2, 4, 6, 8, 10]:
    fc, av, m = ev(R); rows.append((R, fc, av, m))
    print(f"{R:<9}{fc:.3f}      {av:.3f}   {m:.3f}")
best = max(rows, key=lambda r: r[3])
print(f"\nbest relay R={best[0]} -> 곱셈 {best[3]:.3f} (R=0: {rows[0][3]:.3f})")
with open(os.path.join(OUT, "summary_relay_tradeoff.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["R_relay", "compromise", "avail", "곱셈"])
    for r in rows:
        wr.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)])
print("DONE_RELAY")
