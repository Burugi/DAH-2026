"""hier_v2(증거 기반 개선판) vs hier_h2 vs rule — 페어드 유의성 검정.

stance ablation에서 EMERGENCY가 유의하게 해로웠고(p=3.2e-4), ANTI_WORM은 연결성엔
도움·점령엔 해로움이 드러났다. hier_v2는 (1) EMERGENCY→CONTAIN, (2) ANTI_WORM 게이팅을
적용한다. 이 스크립트는 개선이 통계적으로 유의한지 확인한다.

Usage:
    python src/analyze_v2.py
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
                                  paired_t_test, cohens_dz, mean_ci_bootstrap)

BLUES = ["rule", "hier_h2", "hier_v2"]
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

    P("=" * 72)
    P("hier_v2 (증거 기반 개선판) 검증 — %d 페어" % len(keys))
    P("=" * 72)

    P("\n■ 에이전트별 평균 + 95% 부트CI")
    for b in BLUES:
        m, lo, hi = mean_ci_bootstrap(vec(b), seed=1)
        P(f"  {b:<10} {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    P("\n■ 페어드 비교")
    P(f"{'비교':<18} {'평균차':>9} {'95%CI':>20} {'t검정p':>10} {'Wilcoxp':>10} "
      f"{'d_z':>7} {'승률':>7}")
    P("-" * 86)
    for a, b in [("hier_v2", "hier_h2"), ("hier_v2", "rule"), ("hier_h2", "rule")]:
        diff = vec(a) - vec(b)
        lo, hi, _ = bootstrap_ci(diff, seed=2)
        _, _, p_t = paired_t_test(diff)
        _, p_w = wilcoxon_signed_rank(diff)
        P(f"{a+'-'+b:<18} {diff.mean():>+9.4f} [{lo:>+.4f},{hi:>+.4f}] "
          f"{p_t:>10.2e} {p_w:>10.2e} {cohens_dz(diff):>7.3f} "
          f"{float((diff>0).mean()):>6.1%}")

    P("\n■ 위협 클래스별 hier_v2 - hier_h2 (개선판이 어디서 이득/손해?)")
    P(f"{'클래스':<8} {'N':>4} {'평균차':>9} {'95% CI':>22} {'Wilcoxp':>10}")
    P("-" * 58)
    cls_keys = defaultdict(list)
    for k in keys:
        cls_keys[THREAT_CLASS.get(k[0], "기타")].append(k)
    for cls in ["연결성", "가용성", "탐지회피", "점령"]:
        ck = cls_keys[cls]
        diff = np.array([data[k]["hier_v2"] - data[k]["hier_h2"] for k in ck])
        lo, hi, _ = bootstrap_ci(diff, seed=3)
        _, p_w = wilcoxon_signed_rank(diff)
        P(f"{cls:<8} {len(ck):>4} {diff.mean():>+9.4f}   [{lo:>+.4f}, {hi:>+.4f}] "
          f"{p_w:>10.2e}")

    P("\n■ 해석")
    d_v2h = vec("hier_v2") - vec("hier_h2")
    _, p_w = wilcoxon_signed_rank(d_v2h)
    verdict = "유의한 개선" if (d_v2h.mean() > 0 and p_w < 0.05) else \
              "개선 방향이나 비유의" if d_v2h.mean() > 0 else "개선 없음"
    P(f"  · hier_v2 vs hier_h2: {d_v2h.mean():+.4f}, Wilcoxon p={p_w:.2e} → {verdict}")
    d_v2r = vec("hier_v2") - vec("rule")
    _, p_wr = wilcoxon_signed_rank(d_v2r)
    P(f"  · hier_v2 vs rule:    {d_v2r.mean():+.4f}, Wilcoxon p={p_wr:.2e}, "
      f"승률 {float((d_v2r>0).mean()):.1%}")

    with open(os.path.join(OUT_DIR, "v2_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "threat_class"] + BLUES)
        for k in keys:
            w.writerow([k[0], k[1], THREAT_CLASS.get(k[0], "기타")]
                       + [data[k][b] for b in BLUES])
    with open(os.path.join(ROOT, "docs", "hier_v2_validation.md"), "w",
              encoding="utf-8") as f:
        f.write("# hier_v2 검증\n\n```\n" + "\n".join(lines) + "\n```\n")
    print("\n리포트: docs/hier_v2_validation.md")


if __name__ == "__main__":
    print("hier_v2 검증 수집 중 (23 × 15 × 3 = 1035 롤아웃)...")
    data = collect()
    report(data)
