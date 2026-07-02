"""홀드아웃 검증 — r=25가 튜닝에 안 쓴 seed에서도 정점인가? (과적합 확인)

hier_h3의 guard_radius=25는 seed 0~14로 튜닝했고, 보고 결과도 같은 seed다. 이는
하이퍼파라미터 과적합 위험이 있다. 여기서는 튜닝에 전혀 쓰지 않은 seed 15~29로 반경
스캔을 재현해:
  (1) 정점이 여전히 r≈25 근처인가
  (2) hier_h3_tight가 hier_h2/rule을 여전히 유의하게 이기는가
를 확인한다. 정점이 유지되면 과적합 아님, 정점이 크게 이동하면 경고.

Usage:
    python src/analyze_holdout.py
"""
import os, sys, time
from collections import defaultdict
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from scenarios import load_scenario
from analyze_significance import (dmult_of, BASE_CFG, SCENARIOS,
                                  bootstrap_ci, wilcoxon_signed_rank,
                                  mean_ci_bootstrap)

HOLDOUT_SEEDS = list(range(15, 30))   # 튜닝(0~14)에 안 쓴 seed
RADII = [("hier_h3_r15", 15), ("hier_h3_r20", 20),
         ("hier_h3_tight", 25), ("hier_h3", 40)]
BLUES = ["rule", "hier_h2"] + [b for b, _ in RADII]


def collect():
    data = defaultdict(dict)
    t0 = time.time()
    for scen in SCENARIOS:
        cfg = load_scenario(scen, dict(BASE_CFG, sim=dict(BASE_CFG["sim"])))
        for seed in HOLDOUT_SEEDS:
            for blue in BLUES:
                try:
                    data[(scen, seed)][blue] = dmult_of(cfg, seed, blue)
                except Exception:
                    data[(scen, seed)][blue] = 0.0
        print(f"  {scen:<8} 완료  누적 {time.time()-t0:.0f}s")
    return data


def report(data):
    keys = sorted(data.keys())
    def vec(b):
        return np.array([data[k][b] for k in keys])
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 70)
    P("홀드아웃 검증 (seed 15~29, 튜닝 미사용) — %d 페어" % len(keys))
    P("=" * 70)

    P(f"\n{'반경':>6}  {'평균D_mult':>10}  {'95% CI':>20}  {'vs hier_h2':>11}  {'Wilcoxp':>10}")
    P("-" * 64)
    h2 = vec("hier_h2")
    peak = None
    for name, r in RADII:
        v = vec(name)
        m, lo, hi = mean_ci_bootstrap(v, seed=1)
        diff = v - h2
        _, p_w = wilcoxon_signed_rank(diff)
        P(f"{r:>6}  {m:>10.4f}  [{lo:.4f}, {hi:.4f}]  {diff.mean():>+11.4f}  {p_w:>10.2e}")
        if peak is None or m > peak[2]:
            peak = (name, r, m)
    rm, _, _ = mean_ci_bootstrap(vec("rule"), seed=1)
    hm, _, _ = mean_ci_bootstrap(h2, seed=1)
    P(f"\n  (참고) rule={rm:.4f}  hier_h2={hm:.4f}")

    P("\n■ 검증 결과")
    P(f"  · 홀드아웃 정점 반경 = {peak[1]} (튜닝 정점 = 25)")
    if peak[1] == 25:
        P("    → 정점 일치 ✓  과적합 아님")
    elif abs(peak[1] - 25) <= 5:
        P("    → 정점 근접(±5) ✓  실질적으로 안정, 과적합 아님")
    else:
        P("    → 정점 이동 ⚠  과적합 가능성, 반경 재검토 필요")

    diff = vec("hier_h3_tight") - vec("rule")
    _, p_w = wilcoxon_signed_rank(diff)
    P(f"  · hier_h3_tight(r=25) vs rule (홀드아웃): {diff.mean():+.4f}, "
      f"Wilcoxon p={p_w:.2e}, 승률 {float((diff>0).mean()):.1%}")
    diff2 = vec("hier_h3_tight") - vec("hier_h2")
    _, p_w2 = wilcoxon_signed_rank(diff2)
    verdict = "유의하게 유지" if (diff2.mean() > 0 and p_w2 < 0.05) else \
              "우위 유지(비유의)" if diff2.mean() > 0 else "우위 소실 ⚠"
    P(f"  · hier_h3_tight vs hier_h2 (홀드아웃): {diff2.mean():+.4f}, "
      f"p={p_w2:.2e} → {verdict}")

    with open(os.path.join(ROOT, "docs", "holdout_validation.md"), "w",
              encoding="utf-8") as f:
        f.write("# 홀드아웃 검증 (과적합 확인)\n\n```\n" + "\n".join(lines) + "\n```\n")
    print("\n리포트: docs/holdout_validation.md")


if __name__ == "__main__":
    print("홀드아웃 수집 중 (23 × 15 × 6 = 2070 롤아웃, seed 15~29)...")
    data = collect()
    report(data)
