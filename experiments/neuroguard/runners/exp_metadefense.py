# -*- coding: utf-8 -*-
"""B20 four-layer meta-defense vs the SIMULTANEOUS multi-vector attack.
Push the 'adaptive' router further: add VECTOR DETECTION (L2) and graded routing (L3), and
sweep sensing accuracy q to show residual cost falls as detection improves (sensing = lever).

  L1 invariant detection : per-drone compromise / frontier signal (already observable)
  L2 effect classification: detect which vectors are active (worm / availability-attack) with
                            sensing accuracy q  (q=1 perfect, q=.5 coin-flip)
  L3 graded response      : route each drone to the matching specialist counter
                            worm -> predictive path-cut ; availability-attack -> AllowTraffic restore
  L4 online adaptation    : reward-feedback toggle of the availability-mode strength
Compare to the best prior fixed router ('adaptive') on every attack combo + a sensing sweep on
the full [worm+jam+block] attack.
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
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}
VEC_KO = {"W": "웜", "J": "재밍", "B": "차단"}


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            own = int(self.name.split("_")[-1])
            v = self.VECS[own % len(self.VECS)]
            return self._emit(int(self.np_random.choice(VEC_AIDS[v])), obs)
    return ComboRed


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def adaptive_act(i, comp, pos, ml, rising):
    if i in comp: return 3
    fr = frontier(i, comp, pos, ml)
    if rising: return 6 if fr else (4 if comp else 1)
    return 4 if comp else 7


def meta_act(i, comp, pos, ml, det_worm, det_avail, avail_strength):
    """L3 graded routing from L2 detections (victim-side restore variant)."""
    if i in comp: return 3                                   # L1 invariant: clean own infection
    fr = frontier(i, comp, pos, ml)
    if det_worm and comp and fr: return 6                    # worm specialist: cut next hop
    if comp and not fr:                                      # supporting clean drones
        if det_avail and avail_strength > 0.5: return 7      # availability specialist: restore comms
        return 4                                             # else help retake
    return 7 if det_avail else 1                             # idle: restore if under DoS, else monitor


def meta2_act(i, comp, pos, ml, det_worm, det_avail):
    """L3 SOURCE-NEUTRALISATION routing: every vector emanates from a compromised drone, so the
    effective counter is fast retake of the source; AllowTraffic only for truly-idle clean drones."""
    if i in comp: return 3                                   # remove own infection (source kill)
    fr = frontier(i, comp, pos, ml)
    if det_worm and comp and fr: return 6                    # cut the worm's next hop
    if comp: return 4                                        # neutralise the attacking source (any vector)
    return 7 if det_avail else 1                             # only idle clean: restore comms / monitor


def rollout(seed, red, mode, vectors, q=1.0):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 12345)
    worm_true = "W" in vectors
    avail_true = ("J" in vectors) or ("B" in vectors)
    hist = []; rsum = 0.0
    avail_strength = 0.5; prev_r = None                       # L4 online adaptation state
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        # L2 effect classification with sensing accuracy q
        det_worm = worm_true if rng.random() < q else (not worm_true)
        det_avail = avail_true if rng.random() < q else (not avail_true)
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if mode == "adaptive":
                aid = adaptive_act(i, comp, pos, ml, rising)
            elif mode == "meta2":                            # B20 source-neutralisation routing
                aid = meta2_act(i, comp, pos, ml, det_worm, det_avail)
            else:                                            # meta (B20) victim-restore routing
                aid = meta_act(i, comp, pos, ml, det_worm, det_avail, avail_strength)
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        r = float(np.mean(list(rew.values()))) if rew else 0.0
        rsum += r
        # L4: if availability mode coincided with reward improvement, strengthen it; else relax
        if mode == "meta" and prev_r is not None and det_avail:
            avail_strength = float(np.clip(avail_strength + (0.1 if r > prev_r else -0.1), 0.0, 1.0))
        prev_r = r
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, rsum


def ev(red, mode, vectors, q=1.0):
    rs = [rollout(s, red, mode, vectors, q) for s in EVAL]
    return float(np.mean([c for c, _ in rs])), float(np.mean([r for _, r in rs]))


# ---- (1) sensing sweep on the FULL simultaneous attack ----
FULL = ["W", "J", "B"]; full_red = make_combo_red(FULL)
QS = [0.5, 0.6, 0.7, 0.85, 1.0]
print("=== B20 meta-defense: sensing accuracy q sweep on FULL [웜+재밍+차단] ===")
adC, adR = ev(full_red, "adaptive", FULL)
print(f"  baseline adaptive(감지 없음): 점령 {adC:.3f}  보상 {adR:.0f}")
mC, mR = [], []      # meta (victim-restore)
m2C, m2R = [], []    # meta2 (source-neutralisation)
for q in QS:
    c, r = ev(full_red, "meta", FULL, q); mC.append(c); mR.append(r)
    c2, r2 = ev(full_red, "meta2", FULL, q); m2C.append(c2); m2R.append(r2)
    print(f"  q={q:.2f}:  meta(victim-restore) {c:.3f}|{r:6.0f}    meta2(source-kill) {c2:.3f}|{r2:6.0f}")

fig, ax1 = plt.subplots(figsize=(8.6, 5.0))
ax1.plot([100*q for q in QS], mC, "o-", color="#e69", lw=1.8, label="meta(victim-restore) 점령")
ax1.plot([100*q for q in QS], m2C, "o-", color="crimson", lw=2.2, label="meta2(source-kill) 점령")
ax1.axhline(adC, ls="--", color="gray", alpha=0.7, label=f"adaptive 점령 {adC:.2f}")
ax1.set_xlabel("L2 벡터 감지 정확도 q (%) — 센싱"); ax1.set_ylabel("최종 점령(낮을수록 좋음)", color="crimson")
ax2 = ax1.twinx()
ax2.plot([100*q for q in QS], m2R, "s-", color="navy", lw=2.2, label="meta2 보상")
ax2.axhline(adR, ls="--", color="navy", alpha=0.5, label=f"adaptive 보상 {adR:.0f}")
ax2.set_ylabel("누적 보상=가용성(높을수록 좋음)", color="navy")
ax1.set_title("B20 메타방어: 감지(센싱)+올바른 응답(source-kill)이라야 잔여비용↓")
l1, la = ax1.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la + lb, fontsize=8, loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig44_metadefense.png"), dpi=130); plt.close()

# ---- (2) meta@q=1 vs adaptive across ALL 7 combos ----
COMBOS = []
for k in (1, 2, 3):
    COMBOS += [list(c) for c in itertools.combinations("WJB", k)]
print("\n=== meta2(q=1, source-kill) vs adaptive across all combos (점령 | 보상) ===")
rows = []
for combo in COMBOS:
    nm = "+".join(VEC_KO[v] for v in combo); red = make_combo_red(combo)
    ac, ar = ev(red, "adaptive", combo)
    mc, mr = ev(red, "meta2", combo, 1.0)
    rows.append((nm, ac, ar, mc, mr))
    win = "✓meta2" if (mc <= ac and mr >= ar) else ("~" if mc <= ac or mr >= ar else "✗")
    print(f"  {nm:10}  adaptive {ac:.3f}|{ar:6.0f}   meta2 {mc:.3f}|{mr:6.0f}   {win}")

with open(os.path.join(OUT, "summary_metadefense.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["combo", "adaptive_점령", "adaptive_보상", "meta2_점령", "meta2_보상"])
    for nm, ac, ar, mc, mr in rows: wr.writerow([nm, round(ac, 3), round(ar, 0), round(mc, 3), round(mr, 0)])
    wr.writerow([])
    wr.writerow(["FULL_sensing_q"] + [str(q) for q in QS])
    wr.writerow(["meta_victim_점령"] + [round(c, 3) for c in mC])
    wr.writerow(["meta2_source_점령"] + [round(c, 3) for c in m2C])
    wr.writerow(["meta2_source_보상"] + [round(r, 0) for r in m2R])
    wr.writerow(["adaptive_baseline", round(adC, 3), round(adR, 0)])

print(f"\nFULL 동시공격: adaptive {adC:.3f}/{adR:.0f}")
print(f"  meta(victim-restore)@q=1 {mC[-1]:.3f}/{mR[-1]:.0f}  | meta2(source-kill)@q=1 {m2C[-1]:.3f}/{m2R[-1]:.0f}")
print("Saved fig44_metadefense.png, summary_metadefense.csv")
