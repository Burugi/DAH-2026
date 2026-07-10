"""graph_sh 자가치유 항 가중치 민감도 — 이식 효과가 평탄 영역인가 knife-edge인가.

graph_sh의 +0.169(@k=6) 개선이 자가치유 가중치의 넓은 평탄 영역에서 재현되면 "이식 가능한
지식" 주장의 재현성이 강해지고, 특정 한 점에서만 나오면 요행이다. selfheal 가중치를 여러 점
흔들어 D_mult가 평탄한지 확인한다. blue_brain 주입으로 재코딩 없이 평가.

산출: results/sh_sensitivity/pairs_long.csv
Usage: python src/run_sh_sensitivity.py --seeds 15 --weights 0,0.25,0.5,1,2,4 --budgets 3,6
"""
import os, sys, csv, time, argparse
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import run as _run
from viz import score
from scenarios import load_scenario
from agents.brains import use_rl
from agents.rl import ensure_trained
from agents.experimental import GraphCentralityBlue
from run_arch_matrix import SCENARIOS, BASE_CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--reds", default="rule,llm,rl")
    ap.add_argument("--weights", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--budgets", default="3,6")
    a = ap.parse_args()

    OUT = os.path.join(ROOT, "results", "sh_sensitivity")
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(a.seeds)); reds = a.reds.split(",")
    weights = [float(w) for w in a.weights.split(",")]
    budgets = [b if b == "inf" else int(b) for b in a.budgets.split(",")]
    base = dict(BASE_CFG, seeds=seeds, red_types=reds, blue_types=["graph_sh"])
    use_rl(*ensure_trained(base, 100))

    total = len(weights)*len(budgets)*len(SCENARIOS)*len(seeds)*len(reds)
    print(f"sh-sensitivity: {total} rollouts  weights={weights} budgets={budgets}\n")
    rows = []; t0 = time.time()
    for k in budgets:
        cfg_cache = {s: load_scenario(s, dict(base, sim=dict(base["sim"]),
                                              recovery_budget=k)) for s in SCENARIOS}
        for w in weights:
            for scen in SCENARIOS:
                cfg = cfg_cache[scen]
                n = cfg["fleet"]["n_uav"] + cfg["fleet"]["n_ugv"]
                for seed in seeds:
                    for red in reds:
                        fl, rw, ro, df = _run.rollout(
                            cfg, seed, red_type=red, blue_type="__gsh__",
                            blue_brain=GraphCentralityBlue(n, n_hubs=3, selfheal=w))
                        d = score.d_mult_single(ro, fl["link_up"][:len(ro)])
                        rows.append([k, w, scen, seed, red, d])
            print(f"  k={k} w={w}  누적 {time.time()-t0:.0f}s")

    path = os.path.join(OUT, "pairs_long.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "selfheal_w", "scenario", "seed", "red", "D_mult"])
        w.writerows(rows)

    for k in budgets:
        print(f"\n=== k={k}: selfheal 가중치별 평균 D_mult ===")
        for wt in weights:
            v = [r[5] for r in rows if r[0] == k and r[1] == wt]
            print(f"  w={wt:<5} {np.mean(v):.4f}")
    print(f"\n-> {os.path.relpath(path, ROOT)}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
