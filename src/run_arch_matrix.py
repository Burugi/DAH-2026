"""아키텍처 × 시나리오 방어 매트릭스.

각 blue 아키텍처를 A1~A21 + A_MV + A_CONN (23종) 각 시나리오에 대해 평가하고,
정본 D_mult(comp_F1 제외, score.d_mult_single)로 방어 성능을 낸다. 공격(red)은
rule/llm/rl로 고정, blue만 교체하는 페어드 설계((시나리오,시드,red) 쌍 고정).

산출:
  results/arch_matrix/pairs_long.csv   행=(scenario,seed,red,blue), D_mult+분해지표
  results/arch_matrix/by_scenario.csv  시나리오 × blue 평균 D_mult (per red + all)
  docs/arch_matrix_report.md           시나리오별 승자 + 요약 표

Usage:
    python src/run_arch_matrix.py                 # 전체
    python src/run_arch_matrix.py --seeds 5       # 빠른 파일럿
    python src/run_arch_matrix.py --blues rule,predictive,graph,rag
"""
import os, sys, csv, time, argparse
from collections import defaultdict
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

SCENARIOS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11",
             "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21",
             "A_MV", "A_CONN"]

THREAT_CLASS = {
    "A_CONN": "연결성", "A_MV": "연결성",
    "A7": "가용성", "A8": "가용성", "A4": "가용성", "A13": "가용성",
    "A3": "탐지회피", "A6": "탐지회피", "A11": "탐지회피", "A19": "탐지회피",
}
for _s in ["A1", "A2", "A5", "A9", "A10", "A12", "A14", "A15", "A16",
           "A17", "A18", "A20", "A21"]:
    THREAT_CLASS[_s] = "점령"

BASE_CFG = {
    "name": "arch_matrix",
    "fleet": {"n_uav": 12, "n_ugv": 6, "grid": 100, "max_link": 40},
    "steps": 40,
    "sim": {"starting_num_red": 3, "red_spawn_rate": 0.20, "max_length_data_links": 40},
    "defense": {"detector": "multisensor", "snr_thresh": 6,
                "gps_thresh": 8, "response": "safe_mode"},
    "env": "B",
}

OUT_DIR = os.path.join(ROOT, "results", "arch_matrix")
os.makedirs(OUT_DIR, exist_ok=True)


def metrics_of(cfg, seed, red, blue):
    """단일 롤아웃의 정본 D_mult + 분해지표 + 해석용 per-step 지표.

    반환:
      D_mult/availability/final_comp/comp_auc  — 점수 축
      ttc     time-to-containment: 감염 피크(증가 정지) 스텝. 낮을수록 빠른 봉쇄
      peak    피크 감염 비율
      reinfect 재감염 횟수(탈환 후 다시 감염된 드론-스텝 수)
      retake_ct RetakeSuspicious(4) 총 사용 수 (액션 효율 분모)
      hist    액션 0-9 사용 총합 히스토그램 (정책 지문/커버리지 분석용)
    """
    fleet, reward, red_owned, defence = _run.rollout(cfg, seed, red_type=red, blue_type=blue)
    T = len(reward); n = fleet["n"]
    lu = fleet["link_up"][:T]
    V = score.availability(red_owned, lu)
    fc = float(red_owned[-1].sum() / n)
    cauc = float(red_owned.sum(1).mean() / n)
    dm = score.d_mult_single(red_owned, lu)

    traj = red_owned.sum(1)                          # (T,) 스텝별 감염 수
    peak = float(traj.max() / n)
    ttc = int(traj.argmax())                         # 피크 도달(=봉쇄 시작) 스텝
    # 재감염: 드론별 1→0(탈환) 이후 0→1(재감염) 발생 수
    reinf = 0
    for d in range(n):
        col = red_owned[:, d]
        recovered = False
        for t in range(1, T):
            if col[t-1] == 1 and col[t] == 0:
                recovered = True
            elif col[t-1] == 0 and col[t] == 1 and recovered:
                reinf += 1
    bc = defence.get("blue_cnt")                     # (T,10) 스텝별 액션 카운트
    hist = bc.sum(0).astype(int).tolist() if bc is not None else [0]*10
    retake = int(hist[4]) if len(hist) > 4 else 0

    return {"D_mult": dm, "availability": round(V, 4),
            "final_comp": round(fc, 4), "comp_auc": round(cauc, 4),
            "ttc": ttc, "peak": round(peak, 4), "reinfect": reinf,
            "retake_ct": retake, "hist": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--reds", default="rule,llm,rl")
    ap.add_argument("--blues", default="rule,predictive,graph,rag")
    ap.add_argument("--episodes", type=int, default=100, help="rl 학습 에피소드")
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    reds = a.reds.split(",")
    blues = a.blues.split(",")
    base = dict(BASE_CFG, seeds=seeds, red_types=reds, blue_types=blues)

    if "rl" in reds or "rl" in blues:
        use_rl(*ensure_trained(base, a.episodes))

    print(f"arch matrix: {len(SCENARIOS)} scen × {len(seeds)} seed × "
          f"{len(reds)} red × {len(blues)} blue = "
          f"{len(SCENARIOS)*len(seeds)*len(reds)*len(blues)} rollouts")
    print(f"  reds={reds}  blues={blues}\n")

    rows = []          # long format
    t0 = time.time()
    for scen in SCENARIOS:
        cfg = load_scenario(scen, dict(base, sim=dict(base["sim"])))
        cls = THREAT_CLASS.get(scen, "기타")
        for seed in seeds:
            for red in reds:
                for blue in blues:
                    try:
                        m = metrics_of(cfg, seed, red, blue)
                    except Exception as e:
                        m = {"D_mult": 0.0, "availability": 0.0,
                             "final_comp": 1.0, "comp_auc": 1.0,
                             "ttc": 0, "peak": 1.0, "reinfect": 0,
                             "retake_ct": 0, "hist": [0]*10}
                    rows.append([scen, cls, seed, red, blue,
                                 m["D_mult"], m["availability"],
                                 m["final_comp"], m["comp_auc"],
                                 m["ttc"], m["peak"], m["reinfect"],
                                 m["retake_ct"]] + m["hist"])
        print(f"  {scen:<8} 완료  누적 {time.time()-t0:.0f}s")

    # ── long CSV ──────────────────────────────────────────────────────────────
    long_path = os.path.join(OUT_DIR, "pairs_long.csv")
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "threat_class", "seed", "red", "blue",
                    "D_mult", "availability", "final_comp", "comp_auc",
                    "ttc", "peak", "reinfect", "retake_ct"]
                   + [f"act{i}" for i in range(10)])
        w.writerows(rows)

    # ── 시나리오 × blue 평균 (all reds + per red) ──────────────────────────────
    def mean_dm(pred):
        vals = [r[5] for r in rows if pred(r)]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    # 시나리오 × 아키텍처, 셀 하나당 D_mult 하나(전 red·시드 평균). 마지막 MEAN 행.
    by_path = os.path.join(OUT_DIR, "by_scenario.csv")
    with open(by_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "threat_class"] + blues)
        for scen in SCENARIOS:
            cls = THREAT_CLASS.get(scen, "기타")
            w.writerow([scen, cls] +
                       [mean_dm(lambda r, bl=blue: r[0] == scen and r[4] == bl)
                        for blue in blues])
        w.writerow(["MEAN", "-"] +
                   [round(float(np.mean([mean_dm(lambda r, s=scen, bl=blue:
                                                 r[0] == s and r[4] == bl)
                                         for scen in SCENARIOS])), 4) for blue in blues])

    # ── 콘솔 요약: 시나리오 × blue (all-red 평균) + 승자 ──────────────────────
    print("\n" + "=" * 72)
    print("시나리오별 방어 D_mult (all-red 평균)  —  ★=시나리오 승자")
    print("=" * 72)
    print(f"{'scen':<8}{'class':<8}" + "".join(f"{b:>12}" for b in blues) + "   winner")
    print("-" * 72)
    class_acc = defaultdict(lambda: defaultdict(list))
    for scen in SCENARIOS:
        cls = THREAT_CLASS.get(scen, "기타")
        vals = {b: mean_dm(lambda r, bl=b: r[0] == scen and r[4] == bl) for b in blues}
        win = max(vals, key=vals.get)
        cells = "".join(f"{vals[b]:>11.4f}{'★' if b==win else ' '}" for b in blues)
        print(f"{scen:<8}{cls:<8}{cells}   {win}")
        for b in blues:
            class_acc[cls][b].append(vals[b])

    print("\n" + "-" * 72)
    print("위협 클래스별 평균 D_mult")
    print(f"{'class':<10}" + "".join(f"{b:>12}" for b in blues))
    for cls in ["연결성", "가용성", "탐지회피", "점령"]:
        if cls in class_acc:
            print(f"{cls:<10}" + "".join(
                f"{np.mean(class_acc[cls][b]):>12.4f}" for b in blues))

    print(f"\n-> {os.path.relpath(long_path, ROOT)}  ({len(rows)} rows)")
    print(f"-> {os.path.relpath(by_path, ROOT)}")
    print(f"   총 시간: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
