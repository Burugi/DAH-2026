# -*- coding: utf-8 -*-
"""The attacker's dilemma: ACTING EXPOSES. Test the user's insight — attacking reveals the
attacker, so a defender that detects by EXPOSURE (only what red actively touches) turns red's
aggression against it.

  Red TEMPO  : each step a compromised drone ACTS (spread/exploit) with prob=tempo, else lies low
               (Sleep). High tempo = fast spread but loud (exposed). Low tempo = stealthy but slow.
  Defence    : 'perfect' (ground-truth compromise sensor, upper bound) vs 'exposure' (sees a
               compromised drone only once its ACTIVITY is detected, w.p. q*tempo — i.e. you only
               catch what acts). The exposure defender weaponises red's aggression.
Sweep tempo and compare final compromise: does aggression backfire?
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


def make_tempo_red(tempo):
    class TempoRed(brains._Red):
        TEMPO = tempo
        def get_action(self, obs, action_space):
            if self.np_random.uniform() > self.TEMPO:        # lie low to avoid exposure
                return self._emit(0, obs)                     # Sleep (stealth)
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            return self._emit(int(self.np_random.choice([2, 6])), obs)   # exploit / spread (loud)
    return TempoRed


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)


def blue_act(i, comp_vis, pos, ml):
    if i in comp_vis: return 3
    if comp_vis and frontier(i, comp_vis, pos, ml): return 6
    return 4 if comp_vis else 1


def rollout(seed, tempo, defense, q_expose=0.6):
    red = make_tempo_red(tempo)
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 4242)
    detected = set()
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        if defense == "exposure":
            for i in comp:                                    # acting drones expose themselves
                if rng.random() < q_expose * tempo:
                    detected.add(i)
            detected &= comp                                  # forget cleared drones
            comp_vis = comp & detected                        # defender only sees what it caught
        else:
            comp_vis = comp                                   # perfect sensor (upper bound)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        ctx = {"compromised": comp_vis, "ip_to_drone": ip2d, "n": n}
        acts = {a: actions.make_blue_index(blue_act(int(a.split("_")[-1]), comp_vis, pos, ml), env, a, ctx) for a in live}
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n


def ev(tempo, defense): return float(np.mean([rollout(s, tempo, defense) for s in EVAL]))


TEMPOS = [0.2, 0.4, 0.6, 0.8, 1.0]
print("=== attacker tempo vs final compromise (does ACTING expose the attacker?) ===")
print("tempo".ljust(8) + f"{'perfect센서':>14}{'exposure센서':>16}")
perf, expo = [], []
for tp in TEMPOS:
    p = ev(tp, "perfect"); e = ev(tp, "exposure"); perf.append(p); expo.append(e)
    print(f"{tp:<8.1f}{p:14.3f}{e:16.3f}")

# also: a NO-DEFENCE baseline to show raw spread speed by tempo
def raw(seed, tempo):
    red = make_tempo_red(tempo)
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]
    for t in range(cfg["steps"]):
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {a: actions.make_blue_index(1, env, a, {"compromised": set(), "ip_to_drone": ip2d, "n": n}) for a in live}
        _, _, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n
nodef = [float(np.mean([raw(s, tp) for s in EVAL])) for tp in TEMPOS]
print("\n무방어(순수 확산속도):", [f"{v:.2f}" for v in nodef])

plt.figure(figsize=(8.4, 5.0))
plt.plot([100*t for t in TEMPOS], nodef, "o-", color="gray", lw=1.8, label="무방어 (확산만)")
plt.plot([100*t for t in TEMPOS], perf, "s-", color="navy", lw=2, label="완전 센서 방어(상한)")
plt.plot([100*t for t in TEMPOS], expo, "o-", color="crimson", lw=2.4, label="노출기반 능동방어")
plt.xlabel("공격자 공세 속도 tempo (%) — 높을수록 빠르지만 시끄러움")
plt.ylabel("최종 점령 (낮을수록 방어 성공)")
plt.title("공격자의 딜레마: 빨리 칠수록 노출되어 노출기반 방어에 역으로 잡힌다")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig46_exposure.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_exposure.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["tempo", "무방어", "perfect", "exposure"])
    for i, tp in enumerate(TEMPOS): wr.writerow([tp, round(nodef[i], 3), round(perf[i], 3), round(expo[i], 3)])

worst = TEMPOS[int(np.argmax(expo))]
print(f"\n노출기반 방어 기준 공격자 최악 tempo(=방어 최선): {worst}  (그 위/아래로 갈수록 방어가 이김)")
print("Saved fig46_exposure.png, summary_exposure.csv")
