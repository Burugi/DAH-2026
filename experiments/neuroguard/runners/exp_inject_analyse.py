# -*- coding: utf-8 -*-
"""Implement the two MISSING means as synthetic overlays (like our GPS/jam/satellite overlays),
to test the diagnosis from fig44: "residual cost falls only when you add an effective MEANS,
not better detection."

  INJECT (attack overlay) : a compromised drone injects forged commands into a nearby CLEAN drone.
        The injected drone is NOT compromised (invisible to the compromise sensor) but bleeds
        availability every step until cleared. A stealthy availability drain the worm-killer misses.
  ANALYSE (defence overlay): deep-inspect up to `budget` drones/step; an inspected drone that is
        injected is REVEALED and RESET (cleared). This is the sensing+clear MEANS that Inject needs.

Worm containment is held fixed (source-kill retake) so we isolate the Inject/Analyse effect on the
AVAILABILITY axis. Sweep the Analyse budget 0..6 and show availability recovers monotonically —
contrast with fig44 where adding detection WITHOUT a means did nothing.
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
INJ_COST = 2.0          # availability penalty per injected drone per step (overlay)


class WormRed(brains._Red):
    """worm spreader (CybORG-real); injection is applied as an overlay on top of compromise."""
    def get_action(self, obs, action_space):
        if self.mem.get("target") is not None and obs.get("success") is True:
            return self._emit(5, obs)
        return self._emit(int(self.np_random.choice([2, 6])), obs)


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def near_clean(src, comp, injected, pos, n, ml):
    """nearest clean, non-injected drone to a compromised src (overlay inject target)."""
    best, bd = None, 1e9
    for j in range(n):
        if j in comp or j in injected or j == src:
            continue
        d = np.linalg.norm(pos[src] - pos[j])
        if d < ml and d < bd:
            best, bd = j, d
    return best


def rollout(seed, analyse_budget, with_inject=True):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, WormRed); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 999)
    injected = set(); rsum = 0.0; inj_pen = 0.0; scan_ptr = 0
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        injected -= comp                                    # a drone that got fully compromised leaves the inject pool

        # --- ATTACK overlay: each compromised drone injects one nearby clean drone ---
        if with_inject:
            for src in comp:
                tgt = near_clean(src, comp, injected, pos, n, ml)
                if tgt is not None and rng.random() < 0.5:
                    injected.add(tgt)

        # --- DEFENCE overlay: Analyse inspects `budget` drones round-robin, clears injected ones ---
        if analyse_budget > 0:
            for _ in range(analyse_budget):
                d = scan_ptr % n; scan_ptr += 1
                if d in injected:
                    injected.discard(d)                     # revealed + reset

        inj_pen += INJ_COST * len(injected)                 # availability bleed this step

        # --- worm containment (fixed source-kill routing) ---
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in comp:
                aid = 3
            elif comp and frontier(i, comp, pos, ml):
                aid = 6
            elif comp:
                aid = 4
            else:
                aid = 1
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        rsum += float(np.mean(list(rew.values()))) if rew else 0.0
        if all(done.values()):
            break
    comp_final = len(run.compromised_drones(cyborg, n)) / n
    return comp_final, rsum - inj_pen, inj_pen               # availability = sim reward minus inject bleed


def ev(budget, with_inject=True):
    rs = [rollout(s, budget, with_inject) for s in EVAL]
    return (float(np.mean([c for c, _, _ in rs])),
            float(np.mean([r for _, r, _ in rs])),
            float(np.mean([p for _, _, p in rs])))


# reference: worm-only, no inject (the floor we'd have if Inject didn't exist)
c0, r0, _ = ev(0, with_inject=False)
print(f"=== reference (웜만, Inject 없음): 점령 {c0:.3f}  가용성보상 {r0:.0f} ===\n")

BUDGETS = [0, 1, 2, 3, 4, 6]
print("=== Inject 추가 + Analyse 예산 스윕 (means가 잔여비용을 줄이는가?) ===")
print("analyse예산   점령      가용성보상   inject피해")
Cs, Rs, Ps = [], [], []
for b in BUDGETS:
    c, r, p = ev(b, with_inject=True); Cs.append(c); Rs.append(r); Ps.append(p)
    tag = " <- means 없음(fig44 상황)" if b == 0 else ""
    print(f"  {b:>2}        {c:.3f}     {r:8.0f}    {p:7.0f}{tag}")

plt.figure(figsize=(8.4, 5.0))
ax1 = plt.gca()
ax1.plot(BUDGETS, Rs, "o-", color="seagreen", lw=2.3, label="가용성 보상 (Analyse 적용)")
ax1.axhline(r0, ls="--", color="gray", label=f"Inject 없을 때 상한 {r0:.0f}")
ax1.axhline(Rs[0], ls=":", color="crimson", label=f"means 없음(예산0) {Rs[0]:.0f}")
ax1.set_xlabel("Analyse 예산 (step당 점검·정화 드론 수) — *수단*"); ax1.set_ylabel("가용성 보상 (↑좋음)")
ax1.set_title("수단(Analyse)을 더하면 Inject 잔여비용이 단조 감소한다 (fig44 진단 검증)")
ax2 = ax1.twinx()
ax2.plot(BUDGETS, Ps, "s--", color="navy", lw=1.6, label="inject 누적피해")
ax2.set_ylabel("inject 누적 가용성 피해 (↓좋음)", color="navy")
l1, la = ax1.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la + lb, fontsize=8.5, loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig45_inject_analyse.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_inject_analyse.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["analyse_budget", "점령", "가용성보상", "inject피해"])
    wr.writerow(["ref_no_inject", round(c0, 3), round(r0, 0), 0])
    for i, b in enumerate(BUDGETS): wr.writerow([b, round(Cs[i], 3), round(Rs[i], 0), round(Ps[i], 0)])

recov = (Rs[-1] - Rs[0]) / max(1.0, (r0 - Rs[0])) * 100
print(f"\n핵심: means 없음 보상 {Rs[0]:.0f} -> Analyse 예산6 보상 {Rs[-1]:.0f} "
      f"(Inject 피해의 {recov:.0f}% 회복; 상한 {r0:.0f})")
print("Saved fig45_inject_analyse.png, summary_inject_analyse.csv")
