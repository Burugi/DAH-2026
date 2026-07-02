"""통계적 유의성 분석 — hier_h2 개선폭이 노이즈 대비 유의한가?

헤드라인 결과 "hier_h2가 rule 대비 D_mult +1.87%"는 시드 간 표준편차(0.08~0.13)가
개선폭보다 커서, 단순 평균 비교로는 신뢰할 수 없다. 이 스크립트는 동일 (시나리오,
시드) 쌍에서 blue 에이전트만 바꿔 페어링된 D_mult를 수집하고, 다음을 계산한다:

  · 페어드 t-검정 (정확한 t-CDF, 정규근사 없음)
  · Wilcoxon 부호순위 검정 (정규근사 + 연속성 보정)
  · 부트스트랩 95% 신뢰구간 + 부트스트랩 p값
  · Cohen's d_z (페어드 효과크기)
  · 승률 (hier_h2 > rule 쌍의 비율)
  · 위협 클래스별 분해

scipy 없이 순수 numpy/math로 구현. 결과는 콘솔 + docs/significance_analysis.md +
results/significance/pairs.csv 로 출력.

Usage:
    python src/analyze_significance.py
"""
import os, sys, csv, math, time
from collections import defaultdict
import numpy as np

SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

import run as _run
from viz import score
from scenarios import load_scenario

# ── 실험 설정 (sweep_final과 동일 조건) ─────────────────────────────────────────
SCENARIOS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11",
             "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21",
             "A_CONN", "A_MV"]
BLUE_TYPES = ["rule", "react", "hier_h2"]
SEEDS = list(range(15))

THREAT_CLASS = {
    "A_CONN": "연결성", "A_MV": "연결성",
    "A7": "가용성", "A8": "가용성", "A4": "가용성", "A13": "가용성",
    "A3": "탐지회피", "A6": "탐지회피", "A11": "탐지회피", "A19": "탐지회피",
}
for _s in ["A1", "A2", "A5", "A9", "A10", "A12", "A14", "A15", "A16",
           "A17", "A18", "A20", "A21"]:
    THREAT_CLASS[_s] = "점령"

BASE_CFG = {
    "name": "sig",
    "fleet": {"n_uav": 12, "n_ugv": 6, "grid": 100, "max_link": 40},
    "steps": 40,
    "seeds": SEEDS,
    "red_types": ["rule"],
    "blue_types": BLUE_TYPES,
    "sim": {"starting_num_red": 3, "red_spawn_rate": 0.20, "max_length_data_links": 40},
    "defense": {"detector": "multisensor", "snr_thresh": 6,
                "gps_thresh": 8, "response": "safe_mode"},
    "env": "B",
}

OUT_DIR = os.path.join(ROOT, "results", "significance")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════ 통계 함수 (scipy 없이) ═══════════

def _betacf(a, b, x, itmax=200, eps=3e-12):
    """정규화 불완전베타 연속분수 (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """정규화 불완전베타 I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_t_test(diff):
    """페어드 t-검정: (t, df, 양측 p). p = I_{df/(df+t^2)}(df/2, 1/2)."""
    d = np.asarray(diff, float)
    n = len(d)
    md = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), n - 1, (0.0 if md != 0 else 1.0)
    t = md / (sd / math.sqrt(n))
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return t, df, p


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(diff):
    """Wilcoxon 부호순위 검정 (정규근사 + 타이보정 + 연속성 보정). (z, 양측 p)."""
    d = np.asarray(diff, float)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    order = np.argsort(np.abs(d))
    absd = np.abs(d)[order]
    signs = np.sign(d)[order]
    # 평균순위 (타이)
    ranks = np.empty(n)
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j < n and absd[j] == absd[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based 평균
        ranks[i:j] = avg_rank
        t = j - i
        if t > 1:
            tie_term += t ** 3 - t
        i = j
    w_plus = ranks[signs > 0].sum()
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    if sigma == 0:
        return 0.0, 1.0
    cc = 0.5 if w_plus > mu else -0.5  # 연속성 보정
    z = (w_plus - mu - cc) / sigma
    p = 2.0 * (1.0 - _phi(abs(z)))
    return z, p


def bootstrap_ci(diff, B=10000, seed=0):
    """페어드 차이의 부트스트랩 95% CI + 양측 부트스트랩 p값."""
    d = np.asarray(diff, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_le = float((means <= 0).mean())
    frac_ge = float((means >= 0).mean())
    p_boot = 2.0 * min(frac_le, frac_ge)
    return float(lo), float(hi), min(1.0, p_boot)


def cohens_dz(diff):
    d = np.asarray(diff, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def mean_ci_bootstrap(vals, B=10000, seed=0):
    """단일 표본 평균의 부트스트랩 95% CI."""
    v = np.asarray(vals, float)
    n = len(v)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


# ══════════════════════════════════════════════ 데이터 수집 ═══════════════════════

def dmult_of(cfg, seed, blue_type):
    """단일 롤아웃의 D_mult (시드별 정확값, comp_F1 포함)."""
    fleet, reward, red_owned, defence = _run.rollout(
        cfg, seed, red_type="rule", blue_type=blue_type)
    T = len(reward)
    n = fleet["n"]
    V = score.availability(red_owned, fleet["link_up"][:T])
    fc = float(red_owned[-1].sum() / n)
    cauc = float(red_owned.sum(1).mean() / n)
    cf1 = defence["metrics"].get("comp", {}).get("F1", 0.0)
    D_core = float(np.mean([1 - fc, 1 - cauc, cf1]))
    return round(D_core * V, 4)


def collect():
    """(scenario, seed) -> {blue: D_mult} 딕셔너리."""
    data = defaultdict(dict)
    t0 = time.time()
    for scen in SCENARIOS:
        cfg = load_scenario(scen, dict(BASE_CFG, sim=dict(BASE_CFG["sim"])))
        for seed in SEEDS:
            for blue in BLUE_TYPES:
                try:
                    data[(scen, seed)][blue] = dmult_of(cfg, seed, blue)
                except Exception:
                    data[(scen, seed)][blue] = 0.0
        done = [k for k in data if k[0] == scen]
        print(f"  {scen:<8} 완료 ({len(done)} seeds)  누적 {time.time()-t0:.0f}s")
    return data


# ══════════════════════════════════════════════ 분석 + 리포트 ════════════════════

def report(data):
    keys = sorted(data.keys(), key=lambda k: (k[0], k[1]))
    # 페어드 벡터
    def vec(blue):
        return np.array([data[k][blue] for k in keys])

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 70)
    P("통계적 유의성 분석 — D_mult (23 시나리오 × 15 seeds = %d 페어)" % len(keys))
    P("=" * 70)

    # 1) 에이전트별 평균 + 부트스트랩 CI
    P("\n■ 에이전트별 D_mult 평균 + 95% 부트스트랩 CI")
    P(f"{'에이전트':<10} {'평균':>8} {'표준편차':>9} {'95% CI':>22}")
    P("-" * 52)
    for b in BLUE_TYPES:
        v = vec(b)
        m, lo, hi = mean_ci_bootstrap(v, seed=1)
        P(f"{b:<10} {m:>8.4f} {v.std(ddof=1):>9.4f}   [{lo:.4f}, {hi:.4f}]")

    # 2) 페어드 비교
    comparisons = [("hier_h2", "rule"), ("hier_h2", "react"), ("react", "rule")]
    P("\n■ 페어드 비교 (A - B): 검정 3종 + 효과크기 + 승률")
    P(f"{'비교':<18} {'평균차':>9} {'95%CI':>20} {'t검정p':>10} "
      f"{'Wilcoxp':>10} {'d_z':>7} {'승률':>7}")
    P("-" * 88)
    for a, b in comparisons:
        diff = vec(a) - vec(b)
        md = diff.mean()
        lo, hi, p_boot = bootstrap_ci(diff, seed=2)
        _, _, p_t = paired_t_test(diff)
        _, p_w = wilcoxon_signed_rank(diff)
        dz = cohens_dz(diff)
        win = float((diff > 0).mean())
        P(f"{a+'-'+b:<18} {md:>+9.4f} [{lo:>+.4f},{hi:>+.4f}] "
          f"{p_t:>10.2e} {p_w:>10.2e} {dz:>7.3f} {win:>6.1%}")

    # 3) 위협 클래스별 hier_h2 - rule
    P("\n■ 위협 클래스별 hier_h2 - rule (페어드)")
    P(f"{'클래스':<8} {'N':>4} {'평균차':>9} {'95% 부트CI':>22} {'Wilcoxp':>10} {'승률':>7}")
    P("-" * 64)
    cls_keys = defaultdict(list)
    for k in keys:
        cls_keys[THREAT_CLASS.get(k[0], "기타")].append(k)
    for cls in ["연결성", "가용성", "탐지회피", "점령"]:
        ck = cls_keys[cls]
        diff = np.array([data[k]["hier_h2"] - data[k]["rule"] for k in ck])
        md = diff.mean()
        lo, hi, _ = bootstrap_ci(diff, seed=3)
        _, p_w = wilcoxon_signed_rank(diff)
        win = float((diff > 0).mean())
        P(f"{cls:<8} {len(ck):>4} {md:>+9.4f}   [{lo:>+.4f}, {hi:>+.4f}] "
          f"{p_w:>10.2e} {win:>6.1%}")

    # 4) 해석
    P("\n■ 해석")
    diff_hr = vec("hier_h2") - vec("rule")
    _, _, p_t = paired_t_test(diff_hr)
    _, p_w = wilcoxon_signed_rank(diff_hr)
    sig = "유의함" if p_w < 0.05 else "유의하지 않음"
    P(f"  · hier_h2 vs rule: 평균차 {diff_hr.mean():+.4f}, "
      f"Wilcoxon p={p_w:.2e} → 통계적으로 {sig} (α=0.05)")
    P(f"  · 승률 {float((diff_hr>0).mean()):.1%}: hier_h2가 "
      f"{int((diff_hr>0).sum())}/{len(diff_hr)} 페어에서 rule을 이김")
    dz = cohens_dz(diff_hr)
    mag = ("무시할 수준" if abs(dz) < 0.2 else "작음" if abs(dz) < 0.5
           else "중간" if abs(dz) < 0.8 else "큼")
    P(f"  · 효과크기 Cohen's d_z={dz:.3f} ({mag}) — "
      f"페어드 설계라 시나리오 간 분산이 상쇄되어 작은 평균차도 검출 가능")

    # CSV 저장
    csv_path = os.path.join(OUT_DIR, "pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "threat_class"] + BLUE_TYPES)
        for k in keys:
            w.writerow([k[0], k[1], THREAT_CLASS.get(k[0], "기타")]
                       + [data[k][b] for b in BLUE_TYPES])

    md_path = os.path.join(ROOT, "docs", "significance_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 통계적 유의성 분석 (D_mult)\n\n```\n")
        f.write("\n".join(lines))
        f.write("\n```\n")
    print(f"\n페어 데이터: {csv_path}")
    print(f"리포트:      {md_path}")


if __name__ == "__main__":
    print("페어드 D_mult 수집 중 (23 × 15 × 3 = 1035 롤아웃)...")
    data = collect()
    report(data)
