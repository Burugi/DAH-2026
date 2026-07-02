# -*- coding: utf-8 -*-
"""Four performance-improvement candidates, evaluated with the TEAM-ALIGNED combined metric
(comp_F1=0.866 from the dashboard) + availability-multiplicative composite + worst-case.

  A_means  : branch-router + anti-jam MEANS (idle clean drones de-jam -> restore availability)
  B_hub    : branch-router + topology hub-priority retake (clear highest-degree compromised first)
  C_multiobj: availability-aware (retake-first, block only when necessary) — optimise D_core x avail
  D_multisig: multi-signal router (branching OR rising -> block sources; + de-jam idle)
Baselines: branch (current best), adaptive.
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
COMP_F1 = 0.866                    # team-aligned (dashboard), not 1.0
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


def degrees(pos0, ml):
    n = len(pos0)
    return np.array([sum(1 for j in range(n) if j != i and np.linalg.norm(pos0[i] - pos0[j]) < ml) for i in range(n)])


def retake_hub_index(env, agent, comp, hubs, ip2d, sleep):
    """RetakeControl index targeting the highest-degree compromised drone, else any."""
    idx = actions.action_index_map(env, agent)
    cand = idx.get("RetakeControl", [])
    best_i, best_deg = None, -1
    for i, ip in cand:
        d = ip2d.get(ip)
        if d in comp and hubs[d] > best_deg:
            best_i, best_deg = i, hubs[d]
    if best_i is not None: return best_i
    return cand[0][0] if cand else sleep


def rollout(seed, red, defense, vectors):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = degrees(pos0, ml)
    hub_thresh = np.percentile(deg, 75)
    comp_frac, avail_frac, hist, new_window = [], [], [], []
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
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, list(env.agent_actions)[0]).get("Sleep", [(0, None)])[0][0] if env.agent_actions else 0
        live = [a for a in env.active_agents if a in env.agent_actions]
        blocks = 0; dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            fr = frontier(i, comp, pos, ml)
            # ---- choose catalog aid by defense ----
            if defense == "adaptive":
                aid = 3 if i in comp else (6 if (rising and fr) else (4 if comp else 7))
            elif defense == "branch":
                aid = 3 if i in comp else ((6 if fr else 4) if (fast and comp) else (8 if (comp and fr) else (4 if comp else 1)))
            elif defense == "A_means":              # branch + anti-jam means on idle clean
                if i in comp: aid = 3
                elif fast and comp: aid = 6 if fr else 4
                elif comp: aid = 8 if fr else 4
                else: aid = ("dejam" if jam_active else 1)
            elif defense == "B_hub":                # branch + hub-priority retake
                if i in comp: aid = 3
                elif fast and comp: aid = 6 if fr else ("hubretake")
                elif comp: aid = 8 if fr else ("hubretake")
                else: aid = 1
            elif defense == "C_multiobj":           # availability-aware: retake-first, minimal block, de-jam
                if i in comp: aid = 3
                elif comp: aid = (6 if (fast and fr) else 4)
                else: aid = ("dejam" if jam_active else 7)
            elif defense == "D_multisig":           # branching OR rising -> block sources; de-jam idle
                threat = fast or rising
                if i in comp: aid = 3
                elif threat and fr and comp: aid = 6
                elif comp: aid = "hubretake"
                else: aid = ("dejam" if jam_active else 1)
            else:
                aid = 1
            # ---- resolve to wrapper index ----
            if aid == "dejam":
                dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)   # CybORG no-op; availability accounting restores
            elif aid == "hubretake":
                acts[a] = retake_hub_index(env, a, comp, deg, ip2d, sleep)
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
    mult = D_core * availability
    return final, availability, D, mult


def ev(red, defense, vectors):
    rs = [rollout(s, red, defense, vectors) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(4))


DEFENSES = ["adaptive", "branch", "A_means", "B_hub", "C_multiobj", "D_multisig"]
print(f"=== improvements (team-aligned comp_F1={COMP_F1}) : 점령 | 가용성 | D | 곱셈종합 ===")
fc_g, av_g, D_g, m_g = {}, {}, {}, {}
for an, red, vec in ATTACKS:
    print(f"\n[{an}]")
    for d in DEFENSES:
        fc, av, D, m = ev(red, d, vec)
        fc_g[(an, d)] = fc; av_g[(an, d)] = av; D_g[(an, d)] = D; m_g[(an, d)] = m
        print(f"   {d:11} 점령 {fc:.3f} | 가용성 {av:.3f} | D {D:.3f} | 곱셈 {m:.3f}")

names = [a[0] for a in ATTACKS]
print("\n=== 리더보드 (공격 전체 평균) — worst-case 점령 & 곱셈 종합 ===")
print("defense".ljust(12) + "평균점령   worst-case   평균D   곱셈종합")
summary = []
for d in DEFENSES:
    avgfc = float(np.mean([fc_g[(a, d)] for a in names]))
    worst = float(max(fc_g[(a, d)] for a in names))
    avgD = float(np.mean([D_g[(a, d)] for a in names]))
    avgm = float(np.mean([m_g[(a, d)] for a in names]))
    summary.append((d, avgfc, worst, avgD, avgm))
    print(f"{d:12}{avgfc:8.3f}{worst:12.3f}{avgD:9.3f}{avgm:10.3f}")

# plot worst-case & 곱셈종합
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
ds = [s[0] for s in summary]
ax1.bar(ds, [s[2] for s in summary], color="#c55"); ax1.set_ylabel("worst-case 점령 (낮을수록↓)")
ax1.set_title("개선안 견고성 (worst-case)"); ax1.tick_params(axis='x', labelrotation=20)
for i, s in enumerate(summary): ax1.text(i, s[2] + 0.005, f"{s[2]:.3f}", ha="center", fontsize=8)
ax2.bar(ds, [s[4] for s in summary], color="#27a"); ax2.set_ylabel("곱셈 종합 (높을수록↑)")
ax2.set_title("개선안 종합점수 (가용성 곱셈)"); ax2.tick_params(axis='x', labelrotation=20)
for i, s in enumerate(summary): ax2.text(i, s[4] + 0.005, f"{s[4]:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig51_improvements.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_improvements.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["defense", "평균점령", "worst_case", "평균D", "곱셈종합"])
    for s in summary: wr.writerow([s[0], round(s[1], 3), round(s[2], 3), round(s[3], 3), round(s[4], 3)])
print("\nSaved fig51_improvements.png, summary_improvements.csv")
