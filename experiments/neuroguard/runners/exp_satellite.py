# -*- coding: utf-8 -*-
"""Satellite-network realism: intermittent connectivity (high latency / disconnection).
The competition env is a CLOUD SATELLITE network; CybORG DroneSwarm is a proxy without satellite
latency/blackouts. We add the defining satellite feature: each step a fraction p of drones are
'blacked out' — the defender CANNOT observe their compromise nor act on them (a sensing/control
gap). Sweep p and measure how the (otherwise optimal) predictive defense degrades. This grounds
our 'sensing is the lever' finding in the satellite domain.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [3000, 3001, 3002, 3003, 3004]
PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


class _R(brains._Red):
    SEQ = None
    def get_action(self, obs, action_space):
        aid = 5 if (self.mem.get("target") is not None and obs.get("success") is True) else int(self.np_random.choice(self.SEQ))
        return self._emit(aid, obs)
class NearRed(_R): SEQ = [2, 2, 6, 10]
class JamRed(_R):  SEQ = [7, 8, 4, 6]
ATTACKS = {"근접확산": NearRed, "재밍+확산": JamRed, "rule웜": brains.RuleRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)

def blue_act(kind, i, comp, pos, ml, rising):
    if i in comp: return 3
    if kind == "flat": return 4 if comp else 1
    fr = frontier(i, comp, pos, ml)
    if kind == "pred": return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "predOODA":
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    return 1


def rollout(seed, red, defense, p):
    """p = fraction of drones in a PERSISTENT satellite-coverage gap (blind spots for the episode)."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 777); hist = []
    k = int(round(p * n))
    black = set(int(x) for x in rng.choice(n, size=k, replace=False)) if k else set()   # persistent coverage gap
    for t in range(cfg["steps"]):
        comp_true = run.compromised_drones(cyborg, n)
        comp_vis = comp_true - black                              # defender only sees connected drones
        ctx = {"compromised": comp_vis, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp_vis)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            aid = 0 if i in black else blue_act(defense, i, comp_vis, pos, ml, rising)   # blacked-out -> no defensive action
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n


def ev(red, p, defense="pred"):
    return float(np.mean([rollout(s, red, defense, p) for s in EVAL]))


print("=== satellite intermittency p vs final compromise (predictive defense) ===")
print("p".ljust(6) + "".join(f"{a:>12}" for a in ATTACKS))
series = {a: [] for a in ATTACKS}
for p in PS:
    line = f"{p:<6.1f}"
    for a, rc in ATTACKS.items():
        v = ev(rc, p); series[a].append(v); line += f"{v:12.3f}"
    print(line)

plt.figure(figsize=(8.5, 4.8))
for a, col in zip(ATTACKS, ["seagreen", "crimson", "navy"]):
    plt.plot([100*p for p in PS], series[a], "o-", color=col, lw=1.9, label=a)
plt.xlabel("간헐 단절률 (% 드론이 매 step 관측·제어 불가) — 위성망 특성")
plt.ylabel("최종 점령 (낮을수록 방어 성공)")
plt.title("위성망 간헐 단절이 예측 방어를 무너뜨린다 (센싱·제어 공백)")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig37_satellite.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_satellite.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["blackout_p"] + list(ATTACKS))
    for i, p in enumerate(PS): wr.writerow([p] + [round(series[a][i], 3) for a in ATTACKS])
print(f"\n단절 0% -> 50%: 근접 {series['근접확산'][0]:.2f}->{series['근접확산'][-1]:.2f}, "
      f"재밍 {series['재밍+확산'][0]:.2f}->{series['재밍+확산'][-1]:.2f}")
print("Saved fig37_satellite.png, summary_satellite.csv")
