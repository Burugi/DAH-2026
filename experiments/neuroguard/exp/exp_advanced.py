# -*- coding: utf-8 -*-
"""Advanced Decide-phase upgrades, team-aligned metric (comp_F1=0.866):
  obs_clf : OBSERVATION ENHANCEMENT — classify attack type (worm/random/jam) from multi-signal
            (branching + dispersion + jam-load) and route to the matching specialist.
  mpc     : SELF-SIMULATION in Decide — at each step run an internal expected-value worm model
            forward k steps for each candidate MODE {retake, block, allow}, and pick the mode
            maximising predicted (1-compromise) x availability (model-predictive control).
Baselines: C_multiobj (best so far), branch.
"""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src")
os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions

OUT = r"C:\workspace\DAH2026_exp"
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
EVAL = [3000, 3001, 3002, 3003, 3004]
COMP_F1 = 0.866; THRESH = 4
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10], "K": [3, 3], "F": [7, 8]}
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


ATTACKS = [("웜", make_combo_red(["W"]), ["W"]), ("ExploitKnown", make_combo_red(["K"]), ["K"]),
           ("재밍", make_combo_red(["J"]), ["J"]), ("FloodAll", make_combo_red(["F"]), ["F"]),
           ("차단", make_combo_red(["B"]), ["B"]), ("동시(W+J+B)", make_combo_red(["W", "J", "B"]), ["W", "J", "B"]),
           ("rule웜", brains.RuleRed, None)]


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return (d < ml) & (d > 0)


def dispersion(comp, pos, ml):
    cs = list(comp)
    if len(cs) < 2: return 0.0
    return float(np.mean([min(np.linalg.norm(pos[a] - pos[b]) for b in cs if b != a) for a in cs])) / ml


# ---- MPC: internal expected-value worm model, pick best mode ----
MODE_PARAM = {  # (spread_scale, retake_rate, block_cost)  per step
    "retake": (1.0, 0.5, 0.05),
    "block":  (0.4, 0.2, 0.40),
    "allow":  (1.0, 0.0, 0.00),
}
def mpc_choose_mode(comp, A, n, beta=0.35, k=3):
    best_mode, best_val = "retake", -1.0
    for mode, (ss, rt, bc) in MODE_PARAM.items():
        p = np.zeros(n)
        for i in comp: p[i] = 1.0
        be = beta * ss
        for _ in range(k):
            newp = p.copy()
            for j in range(n):
                if p[j] < 1.0:
                    prod = float(np.prod(1 - be * p[A[j]])) if A[j].any() else 1.0
                    newp[j] = p[j] + (1 - p[j]) * (1 - prod)
            p = newp * (1 - rt)
        comp_pred = float(p.mean())
        avail_pred = max(0.0, 1 - comp_pred - bc)
        val = (1 - comp_pred) * avail_pred           # multiplicative objective (대회식)
        if val > best_val: best_val, best_mode = val, mode
    return best_mode


def mode_to_aid(mode, i, comp, fr, jam_active):
    if i in comp: return 3
    if mode == "block": return 6 if fr else (4 if comp else 1)
    if mode == "retake": return 4 if comp else ("dejam" if jam_active else 1)
    return 4 if comp else ("dejam" if jam_active else 7)   # allow


def choose_aid(defense, i, comp, pos, ml, fast, disp, jam_active, mpc_mode):
    fr = frontier(i, comp, pos, ml)
    if defense == "C_multiobj":
        if i in comp: return 3
        if comp: return 6 if (fast and fr) else 4
        return "dejam" if jam_active else 7
    if defense == "branch":
        if i in comp: return 3
        if fast and comp: return 6 if fr else 4
        return 8 if (comp and fr) else (4 if comp else 1)
    if defense == "obs_clf":                       # classify attack type, route to specialist
        if i in comp: return 3
        if jam_active and not (fast or disp > 0.5): atype = "jam"
        elif fast or disp > 0.5: atype = "random"
        else: atype = "worm"
        if atype == "jam": return "dejam" if not comp else 4
        if atype == "random": return 6 if (comp and fr) else (4 if comp else 1)
        return 4 if comp else 7                     # worm -> retake-first (availability)
    if defense == "mpc":                            # self-simulation chose mpc_mode
        return mode_to_aid(mpc_mode, i, comp, fr, jam_active)
    return 1


def rollout(seed, red, defense, vectors):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    comp_frac, avail_frac, new_window = [], [], []
    prev = set(run.compromised_drones(cyborg, n)); k = len(vectors) if vectors else 1
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        comp_frac.append(len(comp) / n)
        new = comp - prev; new_window.append(len(new))
        if len(new_window) > 3: new_window.pop(0)
        fast = sum(new_window) >= THRESH; prev = set(comp)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        disp = dispersion(comp, pos, ml)
        red_jam = sum(1 for i in comp if vectors and vectors[i % k] in JAM_VECS); jam_active = red_jam > 0
        mpc_mode = mpc_choose_mode(comp, adjacency(pos, ml), n) if (defense == "mpc" and comp) else "retake"
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        blocks = 0; dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            aid = choose_aid(defense, i, comp, pos, ml, fast, disp, jam_active, mpc_mode)
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
    return final, availability, D_core * availability


def ev(red, defense, vectors):
    rs = [rollout(s, red, defense, vectors) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))


DEFENSES = ["C_multiobj", "branch", "obs_clf", "mpc"]
print(f"=== advanced Decide upgrades (comp_F1={COMP_F1}) : 점령 | 가용성 | 곱셈종합 ===")
fc_g, m_g = {}, {}
for an, red, vec in ATTACKS:
    print(f"\n[{an}]")
    for d in DEFENSES:
        fc, av, m = ev(red, d, vec); fc_g[(an, d)] = fc; m_g[(an, d)] = m
        print(f"   {d:11} 점령 {fc:.3f} | 가용성 {av:.3f} | 곱셈 {m:.3f}")

names = [a[0] for a in ATTACKS]
print("\n=== 리더보드 ===\n" + "defense".ljust(12) + "평균점령   worst-case   곱셈종합")
rows = []
for d in DEFENSES:
    avgfc = float(np.mean([fc_g[(a, d)] for a in names])); worst = float(max(fc_g[(a, d)] for a in names))
    avgm = float(np.mean([m_g[(a, d)] for a in names])); rows.append((d, avgfc, worst, avgm))
    print(f"{d:12}{avgfc:8.3f}{worst:12.3f}{avgm:10.3f}")
with open(os.path.join(OUT, "summary_advanced.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["defense", "평균점령", "worst_case", "곱셈종합"])
    for r in rows: wr.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3)])
print("\nSaved summary_advanced.csv")
