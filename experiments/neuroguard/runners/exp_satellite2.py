# -*- coding: utf-8 -*-
"""Re-validate defense WITH adaptation under satellite coverage gaps.
Coverage gaps (fig37) crush a central-only defense (blind drones become worm reservoirs).
Satellite-appropriate adaptation: each drone runs an AUTONOMOUS local failsafe even while
disconnected from central control (self-detect + self-remove its own infection), and on
RECONNECT the central defense prioritises the previously-blind drones. Compare:
  central-only  : blacked-out drones get no defense (baseline, fig37)
  +failsafe     : disconnected drones self-clean locally
  +failsafe+reconnect : also priority-retake reconnected drones
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
PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


class _R(brains._Red):
    SEQ = None
    def get_action(self, obs, action_space):
        aid = 5 if (self.mem.get("target") is not None and obs.get("success") is True) else int(self.np_random.choice(self.SEQ))
        return self._emit(aid, obs)
class NearRed(_R): SEQ = [2, 2, 6, 10]
ATTACKS = {"근접확산": NearRed, "rule웜": brains.RuleRed}


def frontier(i, comp, pos, ml):
    return any(d != i and np.linalg.norm(pos[i] - pos[d]) < ml for d in comp)

def pred_act(i, comp, pos, ml):
    if i in comp: return 3
    return 8 if (comp and frontier(i, comp, pos, ml)) else (4 if comp else 1)


def rollout(seed, red, p, mode):
    """mode: 'central' | 'failsafe' | 'failsafe_reconnect'. p = persistent coverage-gap fraction."""
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 777); k = int(round(p * n))
    black = set(int(x) for x in rng.choice(n, size=k, replace=False)) if k else set()
    prev_black = set(black)
    for t in range(cfg["steps"]):
        comp_true = run.compromised_drones(cyborg, n)
        comp_vis = comp_true - black
        reconnected = prev_black - black                       # came back into coverage (here: static, so empty)
        ctx = {"compromised": comp_vis, "ip_to_drone": ip2d, "n": n}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in black:
                # disconnected from central control
                if mode in ("failsafe", "failsafe_reconnect") and i in comp_true:
                    aid = 3                                     # AUTONOMOUS local self-clean (no central needed)
                else:
                    aid = 0
            else:
                if mode == "failsafe_reconnect" and i in reconnected and i in comp_true:
                    aid = 3                                     # priority clean a just-reconnected drone
                else:
                    aid = pred_act(i, comp_vis, pos, ml)
            acts[a] = actions.make_blue_index(aid, env, a, ctx)
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg, n)) / n


def ev(red, p, mode): return float(np.mean([rollout(s, red, p, mode) for s in EVAL]))

MODES = [("central-only(기존)", "central"), ("+자율 페일세이프", "failsafe")]
print("=== satellite coverage gap p: central vs autonomous failsafe ===")
for an, rc in ATTACKS.items():
    print(f"\n[{an}]  p" + "".join(f"{m[0]:>18}" for m in MODES))
    series = {m[1]: [] for m in MODES}
    for p in PS:
        line = f"        {p:.1f}"
        for mn, mk in MODES:
            v = ev(rc, p, mk); series[mk].append(v); line += f"{v:18.3f}"
        print(line)
    ATTACKS_RES = ATTACKS  # placeholder
    # plot per attack
    plt.figure(figsize=(7.5, 4.4))
    for (mn, mk), col in zip(MODES, ["crimson", "seagreen"]):
        plt.plot([100*p for p in PS], series[mk], "o-", color=col, lw=2, label=mn)
    plt.xlabel("위성 커버리지 단절률 (%)"); plt.ylabel("최종 점령 (낮을수록 방어 성공)")
    plt.title(f"위성망 적응 방어: 자율 페일세이프가 단절 공백을 메운다 ({an})")
    plt.legend(); plt.tight_layout()
    fn = "fig38_sat_failsafe_near.png" if an == "근접확산" else "fig38_sat_failsafe_worm.png"
    plt.savefig(os.path.join(OUT, fn), dpi=130); plt.close()
    with open(os.path.join(OUT, f"summary_sat_failsafe_{'near' if an=='근접확산' else 'worm'}.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["p"] + [m[0] for m in MODES])
        for i, p in enumerate(PS): wr.writerow([p] + [round(series[m[1]][i], 3) for m in MODES])
    print(f"  -> p=50%: central {series['central'][-1]:.2f} -> failsafe {series['failsafe'][-1]:.2f}")
print("\nSaved fig38_sat_failsafe_*.png + CSVs")
