# -*- coding: utf-8 -*-
"""Adaptive defense v3 — B20 L2 done right: classify the attack by its early EFFECT fingerprint,
then pick the matching best loop (no blind within-episode switching).
Fingerprint (first K steps under a neutral default): leap distance of new infections, compromise
count, live-drone fraction. Offline we record each attack's fingerprint + its oracle-best loop;
online we run K steps, fingerprint, nearest-match -> switch to that loop. Compare to oracle/v1.
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
TRAIN_SEEDS = [4000, 4001, 4002, 4003]      # to build attack fingerprints (offline)
EVAL = [3000, 3001, 3002, 3003, 3004]       # held-out test
LOOPS = ["flat", "pred", "predOODA"]
K = 7                                        # observation window


class _R(brains._Red):
    SEQ = None
    def get_action(self, obs, action_space):
        aid = 5 if (self.mem.get("target") is not None and obs.get("success") is True) else int(self.np_random.choice(self.SEQ))
        return self._emit(aid, obs)
class NearRed(_R): SEQ = [2, 2, 6, 10]
class FarRed(_R):  SEQ = [4, 4, 4, 10]
class RandRed(_R): SEQ = [3, 3, 4, 10]
class JamRed(_R):  SEQ = [7, 8, 4, 6]
class RushRed(_R): SEQ = [2, 4, 6, 10, 10]
class MixRed(_R):  SEQ = [1, 4, 3, 6, 10, 7]
REDS = {"근접": NearRed, "도약": FarRed, "랜덤": RandRed, "재밍": JamRed, "장악러시": RushRed, "혼합": MixRed, "rule": brains.RuleRed, "rl": brains.RLRed}
try:
    from agents.rl import QTable, RED_Q
    brains.use_rl(None, QTable.load(RED_Q))
except Exception:
    pass


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


def rollout(seed, red, default="pred", switch_to=None, switch_at=None):
    """Run; optionally switch loop at step switch_at. Returns (final_comp_frac, fingerprint)."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    hist = []; prev = set(); leap = 0.0; live_frac = []
    for t in range(cfg["steps"]):
        cmp = run.compromised_drones(cyborg, n); ctx = {"compromised": cmp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(cmp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        if t <= K:                                          # accumulate fingerprint signals
            newd = cmp - prev
            for d in newd:
                if prev: leap = max(leap, min(np.linalg.norm(pos[d] - pos[c]) for c in prev))
            live = [a for a in env.active_agents if a in env.agent_actions]
            live_frac.append(len(live) / n)
            fp = (leap / ml, len(cmp) / n, float(np.mean(live_frac)))
        prev = set(cmp)
        kind = default if (switch_at is None or t < switch_at) else switch_to
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(blue_act(kind, int(a.split("_")[-1]), cmp, pos, ml, rising), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n, fp


def comp(red, loop): return float(np.mean([rollout(s, red, default=loop)[0] for s in EVAL]))

# ---- offline: each attack's fingerprint (under neutral pred) + oracle-best loop ----
centroids = []   # (fp_vector, best_loop, name)
print("=== offline fingerprints (first K steps, default=pred) + best loop ===")
for rn, rc in REDS.items():
    fps = np.mean([rollout(s, rc, default="pred")[1] for s in TRAIN_SEEDS], axis=0)
    best = min(LOOPS, key=lambda l: comp(rc, l))
    centroids.append((np.array(fps), best, rn))
    print(f"  {rn:8} fp(leap,comp,live)=({fps[0]:.2f},{fps[1]:.2f},{fps[2]:.2f})  best={best}")

# normalize features for distance
F = np.array([c[0] for c in centroids]); mu = F.mean(0); sd = F.std(0) + 1e-6
def classify(fp):
    z = (np.array(fp) - mu) / sd
    dists = [np.linalg.norm(z - (c[0] - mu) / sd) for c in centroids]
    return centroids[int(np.argmin(dists))][1]

# ---- online: run K steps with pred, fingerprint, switch to classified loop ----
def smart3(seed, red):
    _, fp = rollout(seed, red, default="pred", switch_at=cfg["steps"] + 1)  # full pred just to get fp? -> need fp at K then switch
    loop = classify(fp)
    return rollout(seed, red, default="pred", switch_to=loop, switch_at=K)[0]

print("\nattack".ljust(12) + f"{'oracle':>10}{'SMART v3':>12}{'분류결과':>10}")
rows = []; M = {}
for rn, rc in REDS.items():
    oracle = min(comp(rc, l) for l in LOOPS)
    v3 = float(np.mean([smart3(s, rc) for s in EVAL]))
    # show what it classified (on first eval seed)
    _, fp = rollout(EVAL[0], rc, default="pred", switch_at=cfg["steps"] + 1)
    M[rn] = [oracle, v3]; rows.append([rn, round(oracle, 3), round(v3, 3), classify(fp)])
    print(rn.ljust(12) + f"{oracle:10.3f}{v3:12.3f}{classify(fp):>10}")

print("\n=== worst-case ===")
print(f"  oracle    worst={max(M[r][0] for r in REDS):.3f}")
print(f"  SMART v3  worst={max(M[r][1] for r in REDS):.3f}  (v1 0.22, v2 0.23 대비)")

reds = list(REDS); x = np.arange(len(reds)); w = 0.36
plt.figure(figsize=(11, 5))
plt.bar(x - w/2, [M[r][0] for r in reds], w, label="oracle(즉시 최선)", color="#666666")
plt.bar(x + w/2, [M[r][1] for r in reds], w, label="SMART v3(공격 분류)", color="crimson")
plt.xticks(x, reds, fontsize=8, rotation=15); plt.ylabel("최종 점령 (낮을수록 방어 성공)")
plt.title("적응방어 v3: 공격 지문 분류 → 최적 루프 (B20 L2)")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig35_adaptive_v3.png"), dpi=130); plt.close()
with open(os.path.join(OUT, "summary_adaptive_v3.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack", "oracle", "SMART_v3", "classified"]); wr.writerows(rows)
print("Saved fig35_adaptive_v3.png, summary_adaptive_v3.csv")
