# -*- coding: utf-8 -*-
"""Gap remediation (proxy model): quantify two things CybORG's primitives cannot.
  (1) DECEPTION/DECOY efficacy  — CybORG DeployDecoy is a no-op (observe); here we model a
      honeypot that DIVERTS worm spread, and measure the marginal compromise reduction.
  (2) SWARMFUZZ cascade (A18)   — one GPS-spoofed drone drags neighbours into denser formation,
      adding links and amplifying worm spread (1-point -> swarm chain). Measure amplification.

This is an ANALYTICAL swarm-worm proxy (graph SI model), the same spirit as our synthetic
SNR/GPS overlays and satellite-blackout feature: it complements (does not replace) the CybORG
runs, and is clearly labelled as a proxy. Deterministic (seeded), averaged over seeds.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

OUT = r"C:\workspace\DAH2026_exp"
N = 24            # drones
GRID = 60.0
R = 18.0         # link radius
STEPS = 30
BETA = 0.35      # per-contact infection prob per step
SEEDS = list(range(3000, 3010))


def make_swarm(seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, GRID, size=(N, 2))
    return pos, rng


def neighbors(pos, R):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    A = (d < R) & (d > 0)
    return A


def frontier(inf, A):
    """susceptible nodes adjacent to an infected node."""
    fr = set()
    for i in np.where(inf)[0]:
        for j in np.where(A[i])[0]:
            if not inf[j]:
                fr.add(int(j))
    return fr


def run_episode(seed, decoy_q=0.0, decoy_budget=0, swarmfuzz=False, link_R=R):
    """Graph-SI worm. decoy_q: prob a decoy diverts an infection at a covered frontier node.
    decoy_budget: # frontier nodes blue can cover per step (by degree). swarmfuzz: GPS-spoof drag
    that herds the still-susceptible swarm toward the infected centroid -> bridges sparse gaps."""
    pos, rng = make_swarm(seed)
    pos = pos.copy()
    inf = np.zeros(N, bool)
    seed_node = int(rng.integers(N))
    inf[seed_node] = True
    diverted = np.zeros(N, bool)        # decoy-caught (removed, non-infectious)
    A = neighbors(pos, link_R)
    for t in range(STEPS):
        if swarmfuzz:
            # spoofed formation collapse: susceptibles drift toward the infected centroid,
            # densifying the graph over time so the worm bridges otherwise-disconnected nodes
            cen = pos[inf].mean(axis=0)
            sus = ~inf & ~diverted
            pos[sus] += 0.12 * (cen - pos[sus])
            A = neighbors(pos, link_R)
        fr = frontier(inf, A)
        # blue covers top-degree frontier nodes with decoys
        covered = set()
        if decoy_budget and fr:
            deg = {j: int(A[j].sum()) for j in fr}
            covered = set(sorted(fr, key=lambda j: -deg[j])[:decoy_budget])
        # spread
        new_inf = inf.copy()
        for i in np.where(inf)[0]:
            for j in np.where(A[i])[0]:
                if inf[j] or diverted[j]:
                    continue
                if rng.random() < BETA:
                    if j in covered and rng.random() < decoy_q:
                        diverted[j] = True       # honeypot absorbed the worm; no further spread
                    else:
                        new_inf[j] = True
        inf = new_inf
        if inf.all():
            break
    return inf.sum() / N


def ev(**kw):
    return float(np.mean([run_episode(s, **kw) for s in SEEDS]))


# ---------- (1) Decoy efficacy: sweep interception q, two budgets ----------
QS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
print("=== (1) Decoy/deception efficacy (final infected fraction; lower=better) ===")
print("q\\budget" + "".join(f"{b:>10}" for b in (2, 4, 6)))
series_decoy = {b: [] for b in (2, 4, 6)}
for q in QS:
    line = f"{q:<8.1f}"
    for b in (2, 4, 6):
        v = ev(decoy_q=q, decoy_budget=b); series_decoy[b].append(v); line += f"{v:10.3f}"
    print(line)
base_no_decoy = ev(decoy_q=0.0, decoy_budget=0)
print(f"  no-decoy baseline (CybORG no-op equivalent): {base_no_decoy:.3f}")

plt.figure(figsize=(7.8, 4.6))
for b, col in zip((2, 4, 6), ["#88a", "seagreen", "navy"]):
    plt.plot([100*q for q in QS], series_decoy[b], "o-", lw=2, color=col, label=f"디코이 예산 {b}/step")
plt.axhline(base_no_decoy, ls="--", color="crimson", label=f"디코이 무효(no-op) {base_no_decoy:.2f}")
plt.xlabel("디코이 가로채기 확률 q (%) — 기만 성능"); plt.ylabel("최종 감염 비율 (낮을수록 방어 성공)")
plt.title("기만(디코이) 효과 실측: 작동하는 허니팟이 군집 웜을 흡수한다 (프록시 모델)")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig39_decoy.png"), dpi=130); plt.close()

# ---------- (2) SwarmFuzz cascade amplification (SPARSE swarm so bridging is visible) ----------
SPARSE_R = 11.0   # sparse formation: worm cannot reach everyone without the drag
print(f"\n=== (2) SwarmFuzz cascade (A18): position drag amplifies spread (sparse R={SPARSE_R}) ===")
no_fuzz_nodef = ev(swarmfuzz=False, decoy_budget=0, link_R=SPARSE_R)
fuzz_nodef = ev(swarmfuzz=True, decoy_budget=0, link_R=SPARSE_R)
fuzz_decoy = ev(swarmfuzz=True, decoy_q=0.8, decoy_budget=4, link_R=SPARSE_R)
print(f"  기본(드래그 없음)            : {no_fuzz_nodef:.3f}")
print(f"  SwarmFuzz 드래그(방어 없음)  : {fuzz_nodef:.3f}   (증폭 x{fuzz_nodef/max(no_fuzz_nodef,1e-9):.2f})")
print(f"  SwarmFuzz + 디코이(q=.8,b=4) : {fuzz_decoy:.3f}")

plt.figure(figsize=(7.2, 4.4))
labels = ["기본\n(드래그X)", "SwarmFuzz\n(방어X)", "SwarmFuzz\n+디코이"]
vals = [no_fuzz_nodef, fuzz_nodef, fuzz_decoy]
plt.bar(labels, vals, color=["#88a", "crimson", "seagreen"])
for i, v in enumerate(vals):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
plt.ylabel("최종 감염 비율"); plt.title("SwarmFuzz 연쇄(1대→군집 증폭)와 디코이 완화 (프록시 모델)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig40_swarmfuzz.png"), dpi=130); plt.close()

# CSVs
import csv
with open(os.path.join(OUT, "summary_decoy.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["q"] + [f"budget{b}" for b in (2, 4, 6)])
    for i, q in enumerate(QS):
        w.writerow([q] + [round(series_decoy[b][i], 3) for b in (2, 4, 6)])
with open(os.path.join(OUT, "summary_swarmfuzz.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["case", "infected"])
    w.writerow(["base_no_drag", round(no_fuzz_nodef, 3)])
    w.writerow(["swarmfuzz_no_def", round(fuzz_nodef, 3)])
    w.writerow(["swarmfuzz_decoy", round(fuzz_decoy, 3)])
print("\nSaved fig39_decoy.png, fig40_swarmfuzz.png + CSVs")
