# -*- coding: utf-8 -*-
"""(1) Availability-MULTIPLICATIVE composite (competition-style (A+D)x availability) and
   (2) the 3 new red actions (TargetLeader / ExploitKnown / FloodAll) in single + simultaneous attacks.

Availability now reflects compromise + self-inflicted blocking + RED JAMMING (so FloodAll's DoS is
captured): availability_t = (n - |comp| - blue_blocks - red_jam_targets)/n.
Composite_def = D_core x availability,  D_core = mean(1-점령, 1-AUC, F1)  (availability as multiplier).
New actions map to existing sim primitives, so we also report which class each falls into.
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
COMP_F1 = 1.0
# vector -> red action ids;  L=TargetLeader, K=ExploitKnown, F=FloodAll (new actions, by primitive class)
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10], "L": [2, 6], "K": [3, 3], "F": [7, 8]}
JAM_VECS = {"J", "F"}                    # vectors that degrade availability via DoS


def make_combo_red(vectors):
    class ComboRed(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, action_space):
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            own = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[own % len(self.VECS)]])), obs)
    return ComboRed


# attack: (display, red_factory_or_class, vectors_for_availability)
ATTACKS = [
    ("웜", make_combo_red(["W"]), ["W"]),
    ("재밍", make_combo_red(["J"]), ["J"]),
    ("차단", make_combo_red(["B"]), ["B"]),
    ("동시(W+J+B)", make_combo_red(["W", "J", "B"]), ["W", "J", "B"]),
    ("★리더표적(new)", make_combo_red(["L"]), ["L"]),
    ("★알려진취약점(new)", make_combo_red(["K"]), ["K"]),
    ("★전역재밍FloodAll(new)", make_combo_red(["F"]), ["F"]),
    ("★동시+신규(W+F+L)", make_combo_red(["W", "F", "L"]), ["W", "F", "L"]),
]


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(kind, i, comp, pos, ml, rising):
    if kind == "nodef": return 1
    if i in comp: return 3
    fr = frontier(i, comp, pos, ml)
    if kind == "pred": return 8 if (comp and fr) else (4 if comp else 1)
    if kind == "composite":
        if comp and fr: return 6
        if comp: return 4
        return 7
    if kind == "adaptive":
        if rising: return 6 if fr else (4 if comp else 1)
        return 4 if comp else 7
    return 1


def rollout(seed, red, defense, vectors):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    init = len(run.compromised_drones(cyborg, n))
    k = len(vectors)
    comp_frac, avail_frac, hist = [], [], []
    ttfc = None
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        comp_frac.append(len(comp) / n)
        if ttfc is None and len(comp) > init:
            ttfc = t
        # red jamming load = compromised drones whose lane is a jam vector
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        hist.append(len(comp)); rising = len(hist) >= 3 and hist[-1] > hist[-3]
        live = [a for a in env.active_agents if a in env.agent_actions]
        blocks = 0; acts = {}
        for a in live:
            aid = blue_act(defense, int(a.split("_")[-1]), comp, pos, ml, rising)
            if aid == 6: blocks += 1
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        avail_frac.append(max(0.0, (n - len(comp) - blocks - red_jam) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = comp_frac[-1]; auc = float(np.mean(comp_frac))
    ttfc_n = (ttfc if ttfc is not None else cfg["steps"]) / cfg["steps"]
    availability = float(np.mean(avail_frac))
    A = float(np.mean([final, auc, 1 - ttfc_n, 1 - COMP_F1]))
    D_core = float(np.mean([1 - final, 1 - auc, COMP_F1]))
    D = float(np.mean([1 - final, 1 - auc, COMP_F1, availability]))           # team simple-avg
    comp_mult = D_core * availability                                          # availability as MULTIPLIER
    return final, availability, A, D, comp_mult


def ev(red, defense, vectors):
    rs = [rollout(s, red, defense, vectors) for s in EVAL]
    return tuple(float(np.mean([r[k] for r in rs])) for k in range(5))   # final, avail, A, D, comp_mult


DEFENSES = ["nodef", "pred", "adaptive", "composite"]
print("=== new actions + simultaneous: 점령 | 가용성 | D(단순평균) | 종합곱셈(D_core x avail) ===")
rows = {}
for an, red, vec in ATTACKS:
    print(f"\n[{an}]")
    for d in DEFENSES:
        fc, av, A, D, cm = ev(red, d, vec)
        rows[(an, d)] = (fc, av, A, D, cm)
        print(f"   {d:10} 점령 {fc:.3f} | 가용성 {av:.3f} | D {D:.3f} | 종합곱셈 {cm:.3f}")

print("\n=== 가용성 곱셈 종합점수 — 방어 리더보드 (공격 전체 평균, 높을수록 우수) ===")
defs = ["pred", "adaptive", "composite"]
attack_names = [a[0] for a in ATTACKS]
board = sorted(defs, key=lambda d: -np.mean([rows[(an, d)][4] for an in attack_names]))
for d in board:
    mult = float(np.mean([rows[(an, d)][4] for an in attack_names]))
    Dsimp = float(np.mean([rows[(an, d)][3] for an in attack_names]))
    print(f"   {d:10}  종합곱셈 {mult:.3f}   (단순평균 D {Dsimp:.3f})")

print("\n=== 새 액션 3종 — 단독공격 분류(어느 클래스인가) ===")
for an in ["★리더표적(new)", "★알려진취약점(new)", "★전역재밍FloodAll(new)"]:
    fc, av, A, D, cm = rows[(an, "pred")]
    print(f"   {an:22} (vs pred) 점령 {fc:.3f} | 가용성 {av:.3f}")
print("  -> L/K = 익스플로잇(웜) 클래스(점령형), F = 재밍 클래스(가용성형). 신규 액션은 기존 W/J 벡터로 포섭됨.")

with open(os.path.join(OUT, "summary_newactions_mult.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["attack", "defense", "점령", "가용성", "A", "D_단순평균", "종합곱셈"])
    for an, _, _ in ATTACKS:
        for d in DEFENSES:
            fc, av, A, D, cm = rows[(an, d)]
            wr.writerow([an, d, round(fc, 3), round(av, 3), round(A, 3), round(D, 3), round(cm, 3)])
print("\nSaved summary_newactions_mult.csv")
