"""예산 스윕 분석 — budget_sweep/pairs_long.csv → 표 + 교차 그래프 + 리포트.

핵심 서사: 예산 k가 조이면 할당형(auction/whittle/mincut)이 brute-force(graph)를 이긴다.
  · budget × blue 평균 D_mult 표 + 각 k에서의 승자
  · 할당형 vs graph 페어드 차이 (k별)
  · 교차 그래프: 각 아키텍처의 D_mult(k) 곡선 (crossover 시각화)
  · 효율: whittle이 적은 예산으로 graph(k=inf)를 따라잡는 지점

Usage: python src/analyze_budget.py
"""
import os, sys, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from analyze_significance import wilcoxon_signed_rank, bootstrap_ci

OUT_DIR = os.path.join(ROOT, "results", "budget_sweep")
LONG = os.path.join(OUT_DIR, "pairs_long.csv")
ALLOC = {"auction", "whittle", "mincut"}


def main():
    rows = list(csv.DictReader(open(LONG, encoding="utf-8")))
    blues, budgets = [], []
    for r in rows:
        if r["blue"] not in blues: blues.append(r["blue"])
        if r["budget"] not in budgets: budgets.append(r["budget"])
    # 예산 정렬: 숫자 먼저 오름차순, inf 마지막
    def bkey(b): return (1, 0) if b == "inf" else (0, int(b))
    budgets = sorted(budgets, key=bkey)

    L = []
    def P(s=""): print(s); L.append(s)

    dm = defaultdict(list)            # (budget, blue) -> D_mult list
    pair = defaultdict(dict)         # (budget, scen, seed, red) -> {blue: D_mult}
    for r in rows:
        dm[(r["budget"], r["blue"])].append(float(r["D_mult"]))
        pair[(r["budget"], r["scenario"], r["seed"], r["red"])][r["blue"]] = float(r["D_mult"])

    P("=" * 74)
    P("예산 k × 아키텍처 평균 D_mult  (★=해당 예산 승자, [A]=할당형)")
    P("=" * 74)
    P(f"{'blue':<12}" + "".join(f"{('k='+b):>11}" for b in budgets))
    P("-" * 74)
    win = {}
    for b in budgets:
        best = max(blues, key=lambda bl: np.mean(dm[(b, bl)]))
        win[b] = best
    for bl in blues:
        tag = "[A]" if bl in ALLOC else "   "
        line = f"{tag}{bl:<9}"
        for b in budgets:
            v = np.mean(dm[(b, bl)])
            line += f"{v:>10.3f}{'★' if win[b]==bl else ' '}"
        P(line)
    P("\n각 예산 승자: " + "  ".join(f"k={b}→{win[b]}" for b in budgets))

    # 할당형 vs graph 페어드 (k별)
    P("\n" + "-" * 74)
    P("할당형 vs graph 페어드 차이 (k별) — 양수면 할당형 우세")
    P(f"{'비교':<20}" + "".join(f"{('k='+b):>12}" for b in budgets))
    for alloc in [a for a in ["whittle", "auction", "mincut"] if a in blues]:
        line = f"{alloc+'-graph':<20}"
        for b in budgets:
            keys = [k for k in pair if k[0] == b and alloc in pair[k] and "graph" in pair[k]]
            diff = np.array([pair[k][alloc] - pair[k]["graph"] for k in keys])
            _, p = wilcoxon_signed_rank(diff)
            star = "*" if p < 0.05 else " "
            line += f"{diff.mean():>+11.3f}{star}"
        P(line)

    # 교차 그래프
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs = list(range(len(budgets)))
    for bl in blues:
        ys = [np.mean(dm[(b, bl)]) for b in budgets]
        style = "-o" if bl in ALLOC else "--s"
        lw = 2.4 if bl in ALLOC else 1.3
        ax.plot(xs, ys, style, linewidth=lw, label=("[A] " if bl in ALLOC else "") + bl)
    ax.set_xticks(xs); ax.set_xticklabels([f"k={b}" for b in budgets])
    ax.set_xlabel("active-containment budget"); ax.set_ylabel("mean D_mult")
    ax.set_title("budget vs defense: allocation (solid) overtakes brute-force at tight k")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "budget_crossover.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)

    png_rel = os.path.relpath(os.path.join(OUT_DIR, "budget_crossover.png"), ROOT)
    with open(os.path.join(ROOT, "docs", "budget_sweep_report.md"), "w",
              encoding="utf-8") as f:
        f.write("# 예산 제약 스윕 — 할당 지능이 언제 이기는가\n\n"
                "곡선은 절대 D_mult(k) — 개별 곡선은 단조 증가, 차이가 아니라 절대값 기준.\n\n```\n"
                + "\n".join(L) + "\n```\n\n")
        f.write(f"![crossover](../{png_rel})\n")
    print(f"\n-> docs/budget_sweep_report.md")
    print(f"-> {png_rel}")


if __name__ == "__main__":
    main()
