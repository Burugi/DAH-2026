"""arch_matrix 결과 분석 → 시나리오별 표 + 히트맵 + 페어드 유의성 + 리포트.

run_arch_matrix.py가 만든 results/arch_matrix/pairs_long.csv를 읽어:
  · 시나리오 × blue 평균 D_mult 표 (승자 표시)
  · 위협 클래스 롤업
  · 각 신규 아키텍처 vs rule 페어드 검정 (부트CI, Wilcoxon, 승률)
  · 가용성 vs 감염 축 분해 (왜 이기는가)
  · 히트맵 PNG 2종
  · docs/arch_matrix_report.md

통계함수는 analyze_significance에서 재사용(순수 통계, D_mult 정의와 무관).
D_mult 자체는 pairs_long.csv에 이미 정본(score.d_mult_single)으로 계산돼 있음.

Usage:
    python src/analyze_arch_matrix.py
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

from analyze_significance import bootstrap_ci, wilcoxon_signed_rank, paired_t_test, cohens_dz

OUT_DIR = os.path.join(ROOT, "results", "arch_matrix")
LONG = os.path.join(OUT_DIR, "pairs_long.csv")
CLASS_ORDER = ["연결성", "가용성", "탐지회피", "점령"]


def load():
    rows = list(csv.DictReader(open(LONG, encoding="utf-8")))
    blues, reds, scen_order, cls_of = [], [], [], {}
    for r in rows:
        if r["blue"] not in blues: blues.append(r["blue"])
        if r["red"] not in reds: reds.append(r["red"])
        if r["scenario"] not in scen_order: scen_order.append(r["scenario"])
        cls_of[r["scenario"]] = r["threat_class"]
    return rows, blues, reds, scen_order, cls_of


def _heatmap(ax, M, rows_lbl, cols_lbl, title, cmap="viridis"):
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols_lbl))); ax.set_xticklabels(cols_lbl, rotation=0)
    ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl, fontsize=8)
    ax.set_title(title)
    for i in range(len(rows_lbl)):
        for j in range(len(cols_lbl)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i,j] < 0.6 else "black", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def main():
    rows, blues, reds, scen_order, cls_of = load()
    L = []
    def P(s=""): print(s); L.append(s)

    # 페어드 dict: (scen,seed,red) -> {blue: D_mult}
    pair = defaultdict(dict)
    comp = defaultdict(dict)   # 분해 (availability, final_comp, comp_auc) per blue (평균용)
    for r in rows:
        k = (r["scenario"], r["seed"], r["red"])
        pair[k][r["blue"]] = float(r["D_mult"])
        comp[r["blue"]].setdefault("availability", []).append(float(r["availability"]))
        comp[r["blue"]].setdefault("final_comp", []).append(float(r["final_comp"]))
        comp[r["blue"]].setdefault("comp_auc", []).append(float(r["comp_auc"]))

    # 시나리오 × blue 평균
    def scen_mean(scen, blue):
        v = [float(r["D_mult"]) for r in rows if r["scenario"] == scen and r["blue"] == blue]
        return float(np.mean(v)) if v else float("nan")

    P("=" * 78)
    P(f"아키텍처 × 시나리오 방어 매트릭스  (seeds×reds 평균 D_mult, 정본 comp_F1 제외)")
    P(f"blues={blues}  reds={reds}  scenarios={len(scen_order)}")
    P("=" * 78)
    P(f"\n{'scen':<8}{'class':<8}" + "".join(f"{b:>12}" for b in blues) + "   winner")
    P("-" * 78)
    win_count = defaultdict(int)
    for scen in scen_order:
        vals = {b: scen_mean(scen, b) for b in blues}
        win = max(vals, key=vals.get); win_count[win] += 1
        cells = "".join(f"{vals[b]:>11.4f}{'*' if b==win else ' '}" for b in blues)
        P(f"{scen:<8}{cls_of[scen]:<8}{cells}   {win}")

    # 클래스 롤업
    P("\n" + "-" * 78)
    P("위협 클래스별 평균 D_mult")
    P(f"{'class':<10}" + "".join(f"{b:>12}" for b in blues) + f"{'N_scen':>8}")
    class_scen = defaultdict(list)
    for scen in scen_order: class_scen[cls_of[scen]].append(scen)
    class_M = []
    for cls in CLASS_ORDER:
        if cls not in class_scen: continue
        rowvals = [np.mean([scen_mean(s, b) for s in class_scen[cls]]) for b in blues]
        class_M.append(rowvals)
        P(f"{cls:<10}" + "".join(f"{v:>12.4f}" for v in rowvals) + f"{len(class_scen[cls]):>8}")

    P(f"\n시나리오 승리 횟수: " + "  ".join(f"{b}={win_count[b]}" for b in blues))

    # 페어드 유의성: 신규 vs rule
    P("\n" + "-" * 78)
    P("페어드 유의성 — 각 아키텍처 vs rule  (전체 (scen,seed,red) 페어)")
    P(f"{'비교':<20}{'평균차':>10}{'95%CI':>22}{'Wilcoxp':>11}{'d_z':>8}{'승률':>8}")
    keys = sorted(pair.keys())
    base = "rule"
    for b in blues:
        if b == base: continue
        diff = np.array([pair[k][b] - pair[k][base] for k in keys
                         if b in pair[k] and base in pair[k]])
        lo, hi, _ = bootstrap_ci(diff, seed=1)
        _, p_w = wilcoxon_signed_rank(diff)
        dz = cohens_dz(diff); win = float((diff > 0).mean())
        P(f"{b+'-'+base:<20}{diff.mean():>+10.4f} [{lo:>+.4f},{hi:>+.4f}]"
          f"{p_w:>11.2e}{dz:>8.3f}{win:>7.1%}")

    # 왜 이기는가: 가용성 vs 감염 축
    P("\n" + "-" * 78)
    P("축 분해 (전체 평균) — 가용성↑·감염↓이 좋음")
    P(f"{'blue':<14}{'D_mult':>9}{'availability':>13}{'final_comp':>12}{'comp_auc':>10}")
    for b in blues:
        dm = np.mean([float(r["D_mult"]) for r in rows if r["blue"] == b])
        P(f"{b:<14}{dm:>9.4f}{np.mean(comp[b]['availability']):>13.4f}"
          f"{np.mean(comp[b]['final_comp']):>12.4f}{np.mean(comp[b]['comp_auc']):>10.4f}")

    # ── 모델 × 시나리오 CSV (엑셀 이식용: 행=모델, 열=시나리오) ────────────────
    by_model = os.path.join(OUT_DIR, "by_model.csv")
    with open(by_model, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + scen_order + ["MEAN"])
        for b in blues:
            vals = [round(scen_mean(s, b), 3) for s in scen_order]
            w.writerow([b] + vals + [round(float(np.mean(vals)), 3)])
    print(f"-> results/arch_matrix/by_model.csv  (행=모델 {len(blues)}, 열=시나리오 {len(scen_order)})")

    # ── 히트맵 2종 ────────────────────────────────────────────────────────────
    scenM = np.array([[scen_mean(s, b) for b in blues] for s in scen_order])
    fig, ax = plt.subplots(figsize=(1.6 * len(blues) + 2, 0.34 * len(scen_order) + 1.5))
    _heatmap(ax, scenM, scen_order, blues, "D_mult: 시나리오 × 아키텍처", cmap="viridis")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "heatmap_scenario.png"), dpi=130,
                                    bbox_inches="tight"); plt.close(fig)
    if class_M:
        _ROMAN = {"연결성": "connectivity", "가용성": "availability",
                  "탐지회피": "evasion", "점령": "takeover"}
        fig, ax = plt.subplots(figsize=(1.6 * len(blues) + 2, 0.6 * len(class_M) + 1.6))
        _heatmap(ax, np.array(class_M),
                 [_ROMAN.get(c, c) for c in CLASS_ORDER if c in class_scen],
                 blues, "D_mult: threat-class x architecture", cmap="viridis")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "heatmap_class.png"), dpi=130,
                                        bbox_inches="tight"); plt.close(fig)

    with open(os.path.join(ROOT, "docs", "arch_matrix_report.md"), "w",
              encoding="utf-8") as f:
        f.write("# 아키텍처 × 시나리오 방어 매트릭스 리포트\n\n")
        f.write("정본 D_mult = mean(1-final_comp, 1-comp_auc) × availability "
                "(comp_F1 제외, `score.d_mult_single`).\n\n```\n" + "\n".join(L) + "\n```\n\n")
        f.write("![scenario heatmap](../results/arch_matrix/heatmap_scenario.png)\n\n")
        f.write("![class heatmap](../results/arch_matrix/heatmap_class.png)\n")
    print(f"\n-> docs/arch_matrix_report.md")
    print(f"-> results/arch_matrix/heatmap_scenario.png, heatmap_class.png")


if __name__ == "__main__":
    main()
