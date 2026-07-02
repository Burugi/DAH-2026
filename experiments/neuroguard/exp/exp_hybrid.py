# -*- coding: utf-8 -*-
"""HYBRID defense = C(availability-aware) by default + branch(source-block) only when fast/wide
spread is detected. Goal: keep C's high 곱셈종합 AND branch's low worst-case (best of both).
Team-aligned metric (comp_F1=0.866) + availability-multiplicative composite + worst-case.
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
COMP_F1 = 0.866
THRESH = 4
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10], "K": [3, 3], "L": [2, 6], "F": [7, 8]}
JAM_VECS = {"J", "F"}


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            own = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[own % len(self.VECS)]])), obs)
    return ComboRed


ATTACKS = [("웜", make_combo_red(["W"]), ["W"]),
           ("ExploitKnown", make_combo_red(["K"]), ["K"]),
           ("재밍", make_combo_red(["J"]), ["J"]),
           ("FloodAll", make_combo_red(["F"]), ["F"]),
           ("차단", make_combo_red(["B"]), ["B"]),
           ("동시(W+J+B)", make_combo_red(["W", "J", "B"]), ["W", "J", "B"]),
           ("rule웜", brains.RuleRed, None)]


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def dispersion(comp, pos, ml):
    """mean nearest-infected-neighbour distance among infected / ml. worm=compact(low), random=dispersed(high)."""
    cs = list(comp)
    if len(cs) < 2: return 0.0
    nn = []
    for a in cs:
        d = min(np.linalg.norm(pos[a] - pos[b]) for b in cs if b != a)
        nn.append(d)
    return float(np.mean(nn)) / ml


def choose_aid(defense, i, comp, pos, ml, rising, fast, jam_active, disp):
    fr = frontier(i, comp, pos, ml)
    if defense == "branch":
        if i in comp: return 3
        if fast and comp: return 6 if fr else 4
        return 8 if (comp and fr) else (4 if comp else 1)
    if defense == "C_multiobj":
        if i in comp: return 3
        if comp: return 6 if (fast and fr) else 4
        return "dejam" if jam_active else 7
    if defense == "hybrid":                       # dispersion router: compact->retake(avail), dispersed->block
        if i in comp: return 3
        dispersed = disp > 0.55                    # random/scattered spread -> contain by blocking
        if dispersed and comp: return 6 if fr else 4
        if comp: return 4                          # worm/compact -> retake-first (keep availability)
        return "dejam" if jam_active else 7
    if defense == "adaptive":
        if i in comp: return 3
        if rising: return 6 if fr else (4 if comp else 1)
        return 4 if comp else 7
    return 1


def rollout(seed, red, defense, vectors):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    comp_frac, avail_frac, hist, new_window, disp_trace = [], [], [], [], []
    prev = set(run.compromised_drones(cyborg, n))
    k = len(vectors) if vectors else 1
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        comp_frac.append(len(comp) / n)
        new = comp - prev; new_window.append(len(new))
        if len(new_window) > 3: new_window.pop(0)
        fast = sum(new_window) >= THRESH
        prev = set(comp)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        red_jam = sum(1 for i in comp if vectors and vectors[i % k] in JAM_VECS)
        jam_active = red_jam > 0
        disp = dispersion(comp, pos, ml); disp_trace.append(disp)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        blocks = 0; dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            aid = choose_aid(defense, i, comp, pos, ml, rising, fast, jam_active, disp)
            if aid == "dejam":
                dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else:
                if aid == 6: blocks += 1
                acts[a] = actions.make_blue_index(aid, env, a, ctx)
        restored = min(dejam, red_jam)
        avail_frac.append(max(0.0, (n - len(comp) - blocks - red_jam + restored) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = comp_frac[-1]; auc = float(np.mean(comp_frac)); availability = float(np.mean(avail_frac))
    D_core = float(np.mean([1 - final, 1 - auc, COMP_F1]))
    D = float(np.mean([1 - final, 1 - auc, COMP_F1, availability]))
    return final, availability, D, D_core * availability, float(np.mean(disp_trace))


def ev(red, defense, vectors):
    rs = [rollout(s, red, defense, vectors) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(5))


DEFENSES = ["adaptive", "branch", "C_multiobj", "hybrid"]
print(f"=== HYBRID vs baselines (comp_F1={COMP_F1}) : 점령 | 가용성 | 곱셈종합 ===")
fc_g, m_g = {}, {}
print("(dispersion 진단: worm vs random 분리되는가)")
for an, red, vec in ATTACKS:
    print(f"\n[{an}]")
    for d in DEFENSES:
        fc, av, D, m, disp = ev(red, d, vec)
        fc_g[(an, d)] = fc; m_g[(an, d)] = m
        extra = f" | disp {disp:.2f}" if d == "hybrid" else ""
        print(f"   {d:11} 점령 {fc:.3f} | 가용성 {av:.3f} | 곱셈 {m:.3f}{extra}")

names = [a[0] for a in ATTACKS]
print("\n=== 리더보드 ===")
print("defense".ljust(12) + "평균점령   worst-case   곱셈종합")
rowsum = []
for d in DEFENSES:
    avgfc = float(np.mean([fc_g[(a, d)] for a in names]))
    worst = float(max(fc_g[(a, d)] for a in names))
    avgm = float(np.mean([m_g[(a, d)] for a in names]))
    rowsum.append((d, avgfc, worst, avgm))
    print(f"{d:12}{avgfc:8.3f}{worst:12.3f}{avgm:10.3f}")

plt.figure(figsize=(8.8, 5.0))
ds = [r[0] for r in rowsum]; x = np.arange(len(ds)); w = 0.38
plt.bar(x - w/2, [r[2] for r in rowsum], w, color="#c55", label="worst-case 점령 (↓)")
plt.bar(x + w/2, [r[3] for r in rowsum], w, color="#27a", label="곱셈 종합 (↑)")
for i, r in enumerate(rowsum):
    plt.text(i - w/2, r[2] + 0.005, f"{r[2]:.3f}", ha="center", fontsize=8)
    plt.text(i + w/2, r[3] + 0.005, f"{r[3]:.3f}", ha="center", fontsize=8, fontweight="bold")
plt.xticks(x, ds); plt.ylabel("점수"); plt.legend()
plt.title("하이브리드: C의 곱셈종합 + 분기율 라우터의 worst-case 동시 달성?")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig52_hybrid.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_hybrid.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["defense", "평균점령", "worst_case", "곱셈종합"])
    for r in rowsum: wr.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)])

hy = [r for r in rowsum if r[0] == "hybrid"][0]
print(f"\n핵심: hybrid worst-case {hy[2]:.3f} | 곱셈종합 {hy[3]:.3f}")
print("Saved fig52_hybrid.png, summary_hybrid.csv")
