"""해석·강건성 분석 — pairs_long.csv(확장 지표)에서 보고서용 표/그림 생성.

run_arch_matrix.py가 저장한 확장 컬럼(ttc, peak, reinfect, retake_ct, act0..act9)을 읽어:
  1) 시간/효율 지표표      : D_mult, ttc(봉쇄시간), peak, reinfect, retake_ct, 액션효율
  2) red×blue 성능 행렬    : rule/llm/rl 각 공격자별 D_mult (상호작용 분해)
  3) 위험 지표             : worst-case(min), CVaR@25% (평균이 같아도 꼬리 차이)
  4) 정책 지문(fingerprint): 액션 히스토그램 코사인 유사도 행렬 (아키텍처가 실제로 다른가)
  5) 히트맵 PNG (효율/지문)

Usage:
    python src/analyze_interpret.py
"""
import os, sys, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
OUT_DIR = os.path.join(ROOT, "results", "arch_matrix")
LONG = os.path.join(OUT_DIR, "pairs_long.csv")


def load():
    rows = list(csv.DictReader(open(LONG, encoding="utf-8")))
    blues, reds = [], []
    for r in rows:
        if r["blue"] not in blues: blues.append(r["blue"])
        if r["red"] not in reds: reds.append(r["red"])
    return rows, blues, reds


def cvar(vals, q=0.25):
    """하위 q 분위 평균(worst-case tail). 낮을수록 나쁨."""
    v = np.sort(np.asarray(vals, float))
    k = max(1, int(len(v) * q))
    return float(v[:k].mean())


def main():
    rows, blues, reds = load()
    has_ext = "ttc" in (rows[0] if rows else {})
    L = []
    def P(s=""): print(s); L.append(s)

    by = defaultdict(lambda: defaultdict(list))   # blue -> metric -> list
    hist = defaultdict(lambda: np.zeros(10))       # blue -> action histogram 합
    red_dm = defaultdict(lambda: defaultdict(list))  # blue -> red -> D_mult
    for r in rows:
        b = r["blue"]
        by[b]["D_mult"].append(float(r["D_mult"]))
        red_dm[b][r["red"]].append(float(r["D_mult"]))
        if has_ext:
            by[b]["ttc"].append(float(r["ttc"]))
            by[b]["peak"].append(float(r["peak"]))
            by[b]["reinfect"].append(float(r["reinfect"]))
            by[b]["retake_ct"].append(float(r["retake_ct"]))
            hist[b] += np.array([float(r[f"act{i}"]) for i in range(10)])

    order = sorted(blues, key=lambda b: -np.mean(by[b]["D_mult"]))

    # 1) 시간/효율 지표표
    P("=" * 92)
    P("해석 지표표 (전체 평균)  —  ttc↓ 빠른 봉쇄 · peak↓ · reinfect↓ · 액션효율↑ 좋음")
    P("=" * 92)
    if has_ext:
        P(f"{'blue':<12}{'D_mult':>8}{'ttc':>7}{'peak':>7}{'reinf':>7}"
          f"{'retake':>8}{'효율(D/retake)':>16}")
        P("-" * 66)
        for b in order:
            dm = np.mean(by[b]["D_mult"]); rt = np.mean(by[b]["retake_ct"])
            eff = dm / rt * 100 if rt > 0 else 0.0     # ×100 스케일
            P(f"{b:<12}{dm:>8.3f}{np.mean(by[b]['ttc']):>7.1f}"
              f"{np.mean(by[b]['peak']):>7.3f}{np.mean(by[b]['reinfect']):>7.1f}"
              f"{rt:>8.0f}{eff:>16.3f}")
    else:
        P("  (확장 지표 없음 — run_arch_matrix.py 재실행 필요)")

    # 2) red×blue 성능 행렬
    P("\n" + "-" * 66)
    P("red × blue 성능 행렬 (D_mult) — 공격자 유형별 방어 성능")
    P(f"{'blue':<12}" + "".join(f"{('red='+r):>10}" for r in reds) + f"{'전체':>10}")
    for b in order:
        cells = "".join(f"{np.mean(red_dm[b][r]):>10.3f}" for r in reds)
        P(f"{b:<12}{cells}{np.mean(by[b]['D_mult']):>10.3f}")

    # 3) 위험 지표
    P("\n" + "-" * 66)
    P("위험 지표 — 평균 vs 최악(min) vs CVaR@25% (하위 25% 평균)")
    P(f"{'blue':<12}{'mean':>9}{'min':>9}{'CVaR@25%':>11}")
    for b in order:
        v = by[b]["D_mult"]
        P(f"{b:<12}{np.mean(v):>9.3f}{np.min(v):>9.3f}{cvar(v):>11.3f}")

    # 4) 정책 지문 — 액션 히스토그램 코사인 유사도
    if has_ext:
        P("\n" + "-" * 66)
        P("정책 지문 — 액션분포 코사인 유사도 (1.00=사실상 동일 정책)")
        vecs = {b: hist[b] / (np.linalg.norm(hist[b]) + 1e-9) for b in order}
        P(f"{'':<10}" + "".join(f"{b[:7]:>8}" for b in order))
        sim = np.zeros((len(order), len(order)))
        for i, a in enumerate(order):
            line = f"{a[:9]:<10}"
            for j, c in enumerate(order):
                s = float(vecs[a] @ vecs[c]); sim[i, j] = s
                line += f"{s:>8.2f}"
            P(line)

        # 히트맵: 지문 유사도
        fig, ax = plt.subplots(figsize=(1 + 0.6*len(order), 1 + 0.6*len(order)))
        im = ax.imshow(sim, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        ax.set_title("policy fingerprint (action-dist cosine)")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fingerprint.png"),
                                        dpi=130, bbox_inches="tight"); plt.close(fig)

        # 효율 산점: D_mult vs retake_ct
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for b in order:
            ax.scatter(np.mean(by[b]["retake_ct"]), np.mean(by[b]["D_mult"]), s=40)
            ax.annotate(b, (np.mean(by[b]["retake_ct"]), np.mean(by[b]["D_mult"])),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("retake actions used (mean)"); ax.set_ylabel("D_mult")
        ax.set_title("action efficiency: D_mult vs retake usage"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "efficiency.png"),
                                        dpi=130, bbox_inches="tight"); plt.close(fig)

    with open(os.path.join(ROOT, "docs", "arch_interpret_report.md"), "w",
              encoding="utf-8") as f:
        f.write("# 해석·강건성 분석\n\n```\n" + "\n".join(L) + "\n```\n\n")
        if has_ext:
            f.write("![fingerprint](../results/arch_matrix/fingerprint.png)\n\n")
            f.write("![efficiency](../results/arch_matrix/efficiency.png)\n")
    print("\n-> docs/arch_interpret_report.md")
    if has_ext:
        print("-> results/arch_matrix/fingerprint.png, efficiency.png")


if __name__ == "__main__":
    main()
