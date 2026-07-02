# -*- coding: utf-8 -*-
"""Defending the SIMULTANEOUS multi-vector attack — exhaustive combination sweep.
Attack vectors that can run AT ONCE (assigned per compromised drone by node id):
   W=worm(spread)  J=jam(flood)  B=comms-block.  All 7 non-empty subsets.
Defenses (per-drone loop, both axes):
   flat / pred / predOODA  (single-axis, from prior exp)
   composite : worm-contain (predictive path-cut) + availability-preserve (AllowTraffic on idle clean)
   adaptive  : route by spread signal — rising -> contain mode, else -> availability mode
We measure BOTH axes (점령=compromise, 보상=availability/total-damage proxy) for every
attack-combo x defense, to find a defense that holds the full simultaneous attack on both axes.
"""
import sys, os, itertools
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

VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}     # worm / jam / comms-block action ids
VEC_KO = {"W": "웜", "J": "재밍", "B": "차단"}


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)                      # finish takeover when ready
            own = int(self.name.split("_")[-1])
            v = self.VECS[own % len(self.VECS)]               # SIMULTANEOUS: vector by node id
            return self._emit(int(self.np_random.choice(VEC_AIDS[v])), obs)
    return ComboRed


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(kind, i, comp, pos, ml, rising):
    fr = frontier(i, comp, pos, ml)
    if kind == "flat":
        return 3 if i in comp else (4 if comp else 1)
    if kind == "pred":
        if i in comp: return 3
        return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "predOODA":
        if i in comp: return 3
        if comp and fr: return 6 if rising else 8
        return 4 if comp else 1
    if kind == "composite":                       # both axes at once
        if i in comp: return 3                     # cheap local clean
        if comp and fr: return 6                   # cut the worm's next hop
        if comp: return 4                          # retake remaining compromised
        return 7                                   # idle & clean -> restore/allow traffic (availability)
    if kind == "adaptive":                        # route by spread signal
        if i in comp: return 3
        if rising:                                 # worm spreading -> contain
            return 6 if fr else (4 if comp else 1)
        return 4 if comp else 7                     # contained -> preserve availability
    return 1


def rollout(seed, red, defense):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    hist = []; rsum = 0.0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(defense, int(a.split("_")[-1]), comp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        rsum += float(np.mean(list(rew.values()))) if rew else 0.0
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, rsum


def ev(red, defense):
    rs = [rollout(s, red, defense) for s in EVAL]
    return float(np.mean([c for c, _ in rs])), float(np.mean([r for _, r in rs]))


# all 7 non-empty subsets of W,J,B
COMBOS = []
for k in (1, 2, 3):
    COMBOS += [list(c) for c in itertools.combinations("WJB", k)]
DEFENSES = ["flat", "pred", "predOODA", "composite", "adaptive"]

print("=== attack-combo x defense  (점령 | 보상) ===")
header = "combo".ljust(10) + "".join(f"{d:>18}" for d in DEFENSES)
print(header)
C = {}; R = {}
for combo in COMBOS:
    name = "+".join(VEC_KO[v] for v in combo)
    reds = make_combo_red(combo)
    cs, rs = [], []
    for d in DEFENSES:
        c, r = ev(reds, d); cs.append(c); rs.append(r)
    C[name], R[name] = cs, rs
    print(name.ljust(10) + "".join(f"  {cs[k]:.3f}|{rs[k]:6.0f}" for k in range(len(DEFENSES))))

full = "+".join(VEC_KO[v] for v in ["W", "J", "B"])
# combined score: scoring is (atk+def)*availability -> reward proxies availability, compromise is damage.
# rank defenses on full combo by (low compromise, high reward); print Pareto.
print(f"\n--- FULL 동시공격 [{full}] 방어 순위 ---")
order = sorted(range(len(DEFENSES)), key=lambda k: (C[full][k], -R[full][k]))
for k in order:
    print(f"  {DEFENSES[k]:10}  점령 {C[full][k]:.3f}  보상 {R[full][k]:6.0f}")

# ---- plot: Pareto scatter on the FULL combo (x=compromise lower better, y=reward higher better) ----
plt.figure(figsize=(7.8, 5.2))
for k, d in enumerate(DEFENSES):
    plt.scatter(C[full][k], R[full][k], s=120)
    plt.annotate(d, (C[full][k], R[full][k]), textcoords="offset points", xytext=(7, 4), fontsize=10)
plt.xlabel("최종 점령 (←낮을수록 방어 성공)"); plt.ylabel("누적 보상=가용성 (↑높을수록 방어 성공)")
plt.title(f"동시 멀티벡터[{full}] 방어: 두 축을 함께 잡는 composite/adaptive (실측)")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig42_mv_defense.png"), dpi=130); plt.close()

# ---- heatmap-style: compromise across all combos x defenses ----
fig, ax = plt.subplots(figsize=(8.6, 5.0))
names = list(C.keys())
M = np.array([[C[nm][k] for k in range(len(DEFENSES))] for nm in names])
im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.6)
ax.set_xticks(range(len(DEFENSES))); ax.set_xticklabels(DEFENSES)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for ii in range(len(names)):
    for jj in range(len(DEFENSES)):
        ax.text(jj, ii, f"{M[ii,jj]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("공격조합 × 방어 점령 히트맵 (초록=방어성공)")
fig.colorbar(im, label="최종 점령"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig43_mv_heatmap.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_mv_defense.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["combo"] + [f"{d}_점령" for d in DEFENSES] + [f"{d}_보상" for d in DEFENSES])
    for nm in names: wr.writerow([nm] + [round(v, 3) for v in C[nm]] + [round(v, 0) for v in R[nm]])
print("\nSaved fig42_mv_defense.png, fig43_mv_heatmap.png, summary_mv_defense.csv")
