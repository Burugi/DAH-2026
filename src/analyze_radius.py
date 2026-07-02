"""hier_h3 guard_radius 최적화 — 반경 스펙트럼 스캔.

hier_h3는 반경이 좁을수록(tight) 좋았다(단조 추세). 여기서 반경 5~40을 스캔해 최적점을
찾는다. 반경이 좁을수록 더 많은 healthy 드론이 reserve(Monitor)가 되므로, 극단적으로는
'점령 드론 정리 + 허브 유지 외 전원 감시'에 수렴한다.

Usage:
    python src/analyze_radius.py
"""
import os, sys, time
from collections import defaultdict
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from scenarios import load_scenario
from analyze_significance import (dmult_of, BASE_CFG, SCENARIOS, SEEDS,
                                  bootstrap_ci, wilcoxon_signed_rank,
                                  mean_ci_bootstrap)

RADII = [("hier_h3_r5", 5), ("hier_h3_r10", 10), ("hier_h3_r15", 15),
         ("hier_h3_r20", 20), ("hier_h3_tight", 25), ("hier_h3", 40)]
BLUES = ["rule", "hier_h2"] + [b for b, _ in RADII]


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
    def vec(b):
        return np.array([data[k][b] for k in keys])

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 70)
    P("hier_h3 guard_radius 최적화 — %d 페어" % len(keys))
    P("=" * 70)
    P(f"\n{'반경':>6}  {'평균D_mult':>10}  {'95% CI':>20}  {'vs hier_h2':>11}  {'Wilcoxp':>10}")
    P("-" * 64)
    h2 = vec("hier_h2")
    best = None
    for name, r in RADII:
        v = vec(name)
        m, lo, hi = mean_ci_bootstrap(v, seed=1)
        diff = v - h2
        _, p_w = wilcoxon_signed_rank(diff)
        P(f"{r:>6}  {m:>10.4f}  [{lo:.4f}, {hi:.4f}]  {diff.mean():>+11.4f}  {p_w:>10.2e}")
        if best is None or m > best[2]:
            best = (name, r, m)
    rm, _, _ = mean_ci_bootstrap(vec("rule"), seed=1)
    hm, _, _ = mean_ci_bootstrap(h2, seed=1)
    P(f"\n  (참고) rule={rm:.4f}  hier_h2={hm:.4f}")

    P(f"\n■ 최적 반경 = {best[1]} ({best[0]}), 평균 D_mult = {best[2]:.4f}")
    diff = vec(best[0]) - vec("rule")
    lo, hi, _ = bootstrap_ci(diff, seed=2)
    _, p_w = wilcoxon_signed_rank(diff)
    P(f"  vs rule: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
      f"Wilcoxon p={p_w:.2e} 승률 {float((diff>0).mean()):.1%}")
    P(f"  → rule 대비 개선폭 {(best[2]-rm)/rm*100:.1f}%, "
      f"hier_h2 대비 {(best[2]-hm)/hm*100:.1f}%")

    with open(os.path.join(ROOT, "docs", "radius_optimization.md"), "w",
              encoding="utf-8") as f:
        f.write("# hier_h3 guard_radius 최적화\n\n```\n" + "\n".join(lines) + "\n```\n")
    print("\n리포트: docs/radius_optimization.md")


if __name__ == "__main__":
    print("반경 스캔 수집 중 (23 × 15 × 8 = 2760 롤아웃)...")
    data = collect()
    report(data)
