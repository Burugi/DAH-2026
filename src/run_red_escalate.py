"""Red 권한 상승 스트레스 스윕 — "자가치유 우선" 발견의 유효 경계 검증.

감염 드론이 확률 p로 escalate하면 자가치유(RemoveSessions, id 3)가 무효화되고 재이미징
(RetakeControl, id 4)만 유효하다. p를 올리며 whittle vs whittle_sh vs evo_k6의 방어점수를
관찰: whittle_sh의 우위(자가치유 기반)가 p 상승 시 소멸하고, 탈환 인덱스형(whittle)이
복권되는 교차점을 찾는다. → "어떤 방어 지식이 최적인지는 공격자 강도의 함수".

산출: results/red_escalate/pairs_long.csv  (escalate_p, budget 컬럼 포함)

Usage: python src/run_red_escalate.py --seeds 15 --ps 0,0.3,0.7,1.0 --budgets 3,6
"""
import os, sys, csv, time, argparse
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from run_arch_matrix import SCENARIOS, THREAT_CLASS, BASE_CFG, metrics_of
from scenarios import load_scenario
from agents.brains import use_rl
from agents.rl import ensure_trained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--reds", default="rule,llm,rl")
    ap.add_argument("--blues", default="whittle,whittle_sh,evo_k6")
    ap.add_argument("--ps", default="0,0.3,0.7,1.0")
    ap.add_argument("--budgets", default="3,6")
    a = ap.parse_args()

    OUT = os.path.join(ROOT, "results", "red_escalate")
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(a.seeds)); reds = a.reds.split(","); blues = a.blues.split(",")
    ps = [float(x) for x in a.ps.split(",")]
    budgets = [b if b == "inf" else int(b) for b in a.budgets.split(",")]
    base = dict(BASE_CFG, seeds=seeds, red_types=reds, blue_types=blues)
    use_rl(*ensure_trained(base, 100))

    total = len(ps)*len(budgets)*len(SCENARIOS)*len(seeds)*len(reds)*len(blues)
    print(f"red-escalate sweep: {total} rollouts  ps={ps} budgets={budgets} blues={blues}\n")
    rows = []; t0 = time.time()
    for p in ps:
        for k in budgets:
            for scen in SCENARIOS:
                cfg = load_scenario(scen, dict(base, sim=dict(base["sim"]),
                                               recovery_budget=k, red_escalate_p=p))
                for seed in seeds:
                    for red in reds:
                        for blue in blues:
                            try:
                                m = metrics_of(cfg, seed, red, blue)
                            except Exception:
                                m = {"D_mult": 0.0}
                            rows.append([p, k, scen, THREAT_CLASS.get(scen, "기타"),
                                         seed, red, blue, m["D_mult"]])
            print(f"  p={p} k={str(k):<4} 누적 {time.time()-t0:.0f}s")

    path = os.path.join(OUT, "pairs_long.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["escalate_p", "budget", "scenario", "threat_class", "seed", "red", "blue", "D_mult"])
        w.writerows(rows)

    # 콘솔: p × blue 평균 (k별)
    for k in budgets:
        print(f"\n=== budget k={k}:  escalate_p × blue 평균 D_mult ===")
        print(f"{'blue':<12}" + "".join(f"{('p='+str(p)):>10}" for p in ps))
        for b in blues:
            line = f"{b:<12}"
            for p in ps:
                v = [r[7] for r in rows if r[0] == p and r[1] == k and r[6] == b]
                line += f"{np.mean(v):>10.3f}" if v else f"{'-':>10}"
            print(line)
    print(f"\n-> {os.path.relpath(path, ROOT)}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
