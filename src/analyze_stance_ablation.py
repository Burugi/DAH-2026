"""stance 개별 기여도 분해 (leave-one-stance-out).

우선순위 1(analyze_significance)에서 stance 전환 단독은 역효과(-1.68%)였지만, 이는
전체 stance를 뭉뚱그린 결과다. 여기서는 각 stance(ANTI_JAM/QUARANTINE/ANTI_WORM/
EMERGENCY)를 하나씩 NORMAL로 대체(drop)했을 때 D_mult 하락폭을 측정해 개별 순기여도를
분해한다. 시드별 페어링 + Wilcoxon/부트스트랩으로 유의성까지 확인.

  기여도(stance S) = D_mult(hier_h2 full) - D_mult(drop S)   > 0 이면 도움

Usage:
    python src/analyze_stance_ablation.py
"""
import os, sys, csv, time
from collections import defaultdict
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from scenarios import load_scenario
from analyze_significance import (dmult_of, BASE_CFG, SCENARIOS, SEEDS,
                                  THREAT_CLASS, bootstrap_ci, wilcoxon_signed_rank,
                                  cohens_dz, mean_ci_bootstrap)

DROPS = {
    "ANTI_JAM":   "hier_drop_antijam",
    "QUARANTINE": "hier_drop_quarantine",
    "ANTI_WORM":  "hier_drop_antiworm",
    "EMERGENCY":  "hier_drop_emergency",
}
BLUES = ["rule", "hier_h2"] + list(DROPS.values())

OUT_DIR = os.path.join(ROOT, "results", "significance")
os.makedirs(OUT_DIR, exist_ok=True)


def collect():
    data = defaultdict(dict)
    t0 = time.time()
    for scen in SCENARIOS:
        cfg = load_scenario(scen, dict(BASE_CFG, sim=dict(BASE_CFG["sim"])))
        for seed in SEEDS:
            for blue in BLUES:
                try:
                    data[(scen, seed)][blue] = dmult_of(cfg, seed, blue)
                except Exception:
                    data[(scen, seed)][blue] = 0.0
        print(f"  {scen:<8} 완료  누적 {time.time()-t0:.0f}s")
    return data


def report(data):
    keys = sorted(data.keys())
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    def vec(b):
        return np.array([data[k][b] for k in keys])

    P("=" * 74)
    P("stance 개별 기여도 분해 (leave-one-stance-out, %d 페어)" % len(keys))
    P("=" * 74)

    full = vec("hier_h2")
    m, lo, hi = mean_ci_bootstrap(full, seed=1)
    P(f"\n기준: hier_h2(full) 평균 D_mult = {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    rm, rlo, rhi = mean_ci_bootstrap(vec("rule"), seed=1)
    P(f"      rule 평균 D_mult          = {rm:.4f}  [{rlo:.4f}, {rhi:.4f}]")

    P("\n■ 각 stance 순기여도 = full - drop  (양수면 그 stance가 도움)")
    P(f"{'stance':<12} {'기여도':>9} {'95% 부트CI':>22} {'Wilcoxp':>10} "
      f"{'도움비율':>8}")
    P("-" * 66)
    contrib = {}
    for stance, blue in DROPS.items():
        diff = full - vec(blue)   # full - dropped
        md = diff.mean()
        lo, hi, _ = bootstrap_ci(diff, seed=2)
        _, p_w = wilcoxon_signed_rank(diff)
        help_rate = float((diff > 0).mean())
        contrib[stance] = md
        P(f"{stance:<12} {md:>+9.4f}   [{lo:>+.4f}, {hi:>+.4f}] "
          f"{p_w:>10.2e} {help_rate:>7.1%}")

    # 위협 클래스별 각 stance 기여도
    P("\n■ 위협 클래스별 stance 기여도 (full - drop 평균)")
    cls_keys = defaultdict(list)
    for k in keys:
        cls_keys[THREAT_CLASS.get(k[0], "기타")].append(k)
    header = f"{'클래스':<8}" + "".join(f"{s:>13}" for s in DROPS)
    P(header)
    P("-" * (8 + 13 * len(DROPS)))
    for cls in ["연결성", "가용성", "탐지회피", "점령"]:
        ck = cls_keys[cls]
        row = f"{cls:<8}"
        for stance, blue in DROPS.items():
            diff = np.array([data[k]["hier_h2"] - data[k][blue] for k in ck])
            row += f"{diff.mean():>+13.4f}"
        P(row)

    P("\n■ 해석")
    ranked = sorted(contrib.items(), key=lambda x: -x[1])
    P("  · stance 순기여도 순위: " +
      ", ".join(f"{s}({v:+.4f})" for s, v in ranked))
    best, bestv = ranked[0]
    worst, worstv = ranked[-1]
    P(f"  · 최대 기여: {best} ({bestv:+.4f}) / 최소·역효과: {worst} ({worstv:+.4f})")
    P("  · 우선순위1에서 stance '전환 자체'가 역효과였던 것과 달리, 개별 stance는")
    P("    도움되는 것과 해로운 것이 섞여 있음 → 해로운 stance 제거가 개선 여지")

    csv_path = os.path.join(OUT_DIR, "stance_pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "threat_class"] + BLUES)
        for k in keys:
            w.writerow([k[0], k[1], THREAT_CLASS.get(k[0], "기타")]
                       + [data[k][b] for b in BLUES])
    md_path = os.path.join(ROOT, "docs", "stance_ablation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# stance 개별 기여도 분해\n\n```\n" + "\n".join(lines) + "\n```\n")
    print(f"\n페어 데이터: {csv_path}")
    print(f"리포트:      {md_path}")


if __name__ == "__main__":
    print("stance ablation 수집 중 (23 × 15 × 6 = 2070 롤아웃)...")
    data = collect()
    report(data)
