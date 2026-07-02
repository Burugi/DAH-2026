"""hier_h3 (동적 역할 배분) 검증 — 근접도 기반 dispatch가 hier_h2를 이기는가?

ablation에서 성능 동력이 dispatch(+1.6%)로 확인됐다. hier_h3는 healthy 드론을 위협
근접도로 guardian/reserve 분화한다. 여기서 반경(guard_radius) 3종을 hier_h2·rule과
페어드 비교한다.

Usage:
    python src/analyze_h3.py
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

BLUES = ["rule", "hier_h2", "hier_h3_tight", "hier_h3", "hier_h3_wide"]
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
    P("hier_h3 (동적 역할 배분) 검증 — %d 페어" % len(keys))
    P("=" * 74)

    P("\n■ 에이전트별 평균 + 95% 부트CI")
    for b in BLUES:
        m, lo, hi = mean_ci_bootstrap(vec(b), seed=1)
        P(f"  {b:<15} {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    P("\n■ 각 h3 변형 vs hier_h2 (기준선 대비 개선?)")
    P(f"{'비교':<24} {'평균차':>9} {'95%CI':>20} {'t검정p':>10} {'Wilcoxp':>10} {'승률':>7}")
    P("-" * 84)
    best = None
    for b in ["hier_h3_tight", "hier_h3", "hier_h3_wide"]:
        diff = vec(b) - vec("hier_h2")
        lo, hi, _ = bootstrap_ci(diff, seed=2)
        _, _, p_t = paired_t_test(diff)
        _, p_w = wilcoxon_signed_rank(diff)
        win = float((diff > 0).mean())
        P(f"{b+'-hier_h2':<24} {diff.mean():>+9.4f} [{lo:>+.4f},{hi:>+.4f}] "
          f"{p_t:>10.2e} {p_w:>10.2e} {win:>6.1%}")
        if best is None or diff.mean() > best[1]:
            best = (b, diff.mean())

    P("\n■ 최선 h3 변형 vs rule")
    b = best[0]
    diff = vec(b) - vec("rule")
    lo, hi, _ = bootstrap_ci(diff, seed=4)
    _, p_w = wilcoxon_signed_rank(diff)
    P(f"  {b} vs rule: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
      f"Wilcoxon p={p_w:.2e} 승률 {float((diff>0).mean()):.1%}")

    P("\n■ 위협 클래스별 %s - hier_h2" % b)
    P(f"{'클래스':<8} {'N':>4} {'평균차':>9} {'95% CI':>22} {'Wilcoxp':>10}")
    P("-" * 58)
    cls_keys = defaultdict(list)
    for k in keys:
        cls_keys[THREAT_CLASS.get(k[0], "기타")].append(k)
    for cls in ["연결성", "가용성", "탐지회피", "점령"]:
        ck = cls_keys[cls]
        diff = np.array([data[k][b] - data[k]["hier_h2"] for k in ck])
        lo, hi, _ = bootstrap_ci(diff, seed=5)
        _, p_w = wilcoxon_signed_rank(diff)
        P(f"{cls:<8} {len(ck):>4} {diff.mean():>+9.4f}   [{lo:>+.4f}, {hi:>+.4f}] "
          f"{p_w:>10.2e}")

    P("\n■ 해석")
    diff = vec(b) - vec("hier_h2")
    _, p_w = wilcoxon_signed_rank(diff)
    if diff.mean() > 0 and p_w < 0.05:
        v = "유의한 개선 — 근접도 dispatch가 획일 dispatch를 이김"
    elif diff.mean() > 0:
        v = "개선 방향이나 비유의"
    else:
        v = "개선 없음 — 획일 dispatch(hier_h2)가 이미 충분"
    P(f"  · 최선 {b} vs hier_h2: {diff.mean():+.4f}, p={p_w:.2e} → {v}")

    with open(os.path.join(OUT_DIR, "h3_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "threat_class"] + BLUES)
        for k in keys:
            w.writerow([k[0], k[1], THREAT_CLASS.get(k[0], "기타")]
                       + [data[k][bb] for bb in BLUES])
    with open(os.path.join(ROOT, "docs", "hier_h3_validation.md"), "w",
              encoding="utf-8") as f:
        f.write("# hier_h3 (동적 역할 배분) 검증\n\n```\n" + "\n".join(lines) + "\n```\n")
    print("\n리포트: docs/hier_h3_validation.md")


if __name__ == "__main__":
    print("hier_h3 검증 수집 중 (23 × 15 × 5 = 1725 롤아웃)...")
    data = collect()
    report(data)
