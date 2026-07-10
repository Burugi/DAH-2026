"""예산 제약 스윕 — 능동 봉쇄 예산 k를 바꿔가며 방어점수 측정 (플래그십 실험).

핵심 가설: 현재 환경은 "전원 탈환"이 최적이라 할당 지능이 안 드러난다. 스텝당 능동
봉쇄행동(RemoveSessions/Retake/RetakeRandom/Block = id 3/4/5/6) 수를 k로 상한하면,
"누구를 회복시키나"의 할당 품질이 결정적이 되어 할당형(auction/whittle/mincut)이
분리되는지 확인한다. 예산 초과 시 우선순위 낮은 요청은 Monitor로 강등되며, 우선순위는
브레인의 recovery_priority()가 표현(할당 지능의 통로) — 미제공 브레인은 중립(원순서).

산출: results/budget_sweep/pairs_long.csv  (budget 컬럼 포함)
      콘솔: k × blue D_mult 표

Usage:
    python src/run_budget_sweep.py                       # 기본
    python src/run_budget_sweep.py --seeds 8 --budgets 2,4,6,inf
"""
import os, sys, csv, time, argparse
from collections import defaultdict
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
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reds", default="rule,llm,rl")
    ap.add_argument("--blues", default="rule,graph,hybrid,predictive,auction,whittle,mincut")
    ap.add_argument("--budgets", default="2,4,6,inf")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tag", default="", help="출력 하위폴더 접미사 (기존 결과 보존)")
    a = ap.parse_args()

    OUT_DIR = os.path.join(ROOT, "results", "budget_sweep" + a.tag)
    os.makedirs(OUT_DIR, exist_ok=True)

    seeds = list(range(a.seeds))
    reds = a.reds.split(",")
    blues = a.blues.split(",")
    budgets = [b if b == "inf" else int(b) for b in a.budgets.split(",")]
    base = dict(BASE_CFG, seeds=seeds, red_types=reds, blue_types=blues)
    if "rl" in reds or "rl" in blues:
        use_rl(*ensure_trained(base, a.episodes))

    total = len(SCENARIOS)*len(seeds)*len(reds)*len(blues)*len(budgets)
    print(f"budget sweep: {total} rollouts  budgets={budgets}  blues={blues}\n")

    rows = []
    t0 = time.time()
    for k in budgets:
        for scen in SCENARIOS:
            cfg = load_scenario(scen, dict(base, sim=dict(base["sim"]),
                                           recovery_budget=k))
            cls = THREAT_CLASS.get(scen, "기타")
            for seed in seeds:
                for red in reds:
                    for blue in blues:
                        try:
                            m = metrics_of(cfg, seed, red, blue)
                        except Exception:
                            m = {"D_mult": 0.0, "availability": 0.0, "final_comp": 1.0,
                                 "comp_auc": 1.0, "ttc": 0, "peak": 1.0, "reinfect": 0,
                                 "retake_ct": 0, "hist": [0]*10}
                        rows.append([k, scen, cls, seed, red, blue,
                                     m["D_mult"], m["ttc"], m["peak"], m["retake_ct"]])
            print(f"  k={str(k):<4} {scen:<8} 누적 {time.time()-t0:.0f}s")

    path = os.path.join(OUT_DIR, "pairs_long.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "scenario", "threat_class", "seed", "red", "blue",
                    "D_mult", "ttc", "peak", "retake_ct"])
        w.writerows(rows)

    # 콘솔: budget × blue 평균 D_mult
    print("\n" + "=" * 66)
    print("예산 k × 아키텍처 평균 D_mult  (전 시나리오·시드·red 평균)")
    print("=" * 66)
    print(f"{'blue':<12}" + "".join(f"{('k='+str(k)):>10}" for k in budgets))
    for b in blues:
        line = f"{b:<12}"
        for k in budgets:
            v = [r[6] for r in rows if r[0] == k and r[5] == b]
            line += f"{np.mean(v):>10.3f}" if v else f"{'-':>10}"
        print(line)

    # 각 k에서의 승자
    print("\n각 예산에서의 최고 아키텍처:")
    for k in budgets:
        best, bestv = None, -1
        for b in blues:
            v = np.mean([r[6] for r in rows if r[0] == k and r[5] == b])
            if v > bestv:
                best, bestv = b, v
        print(f"  k={str(k):<4} → {best} ({bestv:.3f})")

    print(f"\n-> {os.path.relpath(path, ROOT)}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
