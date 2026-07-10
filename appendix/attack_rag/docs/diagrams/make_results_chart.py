# -*- coding: utf-8 -*-
"""HVT vs RAG 실험결과 시각화 (18 실공격 시나리오, 5시드 실전조건)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 한글 폰트
for f in ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic"]:
    if any(f.lower() in x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, GRID = "#1f2733", "#5b6875", "#e7ebf0"
TEAL, AMBER, RED, SLATE = "#149e8c", "#e0975a", "#cc6a5c", "#6b7f96"

# ---- 데이터 ----
systems = ["naive RAG\n(개선 전)", "HVT\n(챔피언)", "개선 RAG\n(v2)"]
scores  = [0.599, 0.906, 0.904]
colors1 = [RED, SLATE, TEAL]

scen = [("A19","j",0.042),("A4","j",0.023),("A8","j",0.020),("A1","j",0.019),
        ("A2","j",0.019),("A5","j",0.019),("A9","j",0.019),("A20*","c",0.018),
        ("A-CONN","j",0.019),("A3","j",0.016),("A6","j",0.016),("A7","j",0.016),
        ("A13","j",0.016),("A16","j",0.015),("A10","j",-0.012),("A14","j",-0.012),
        ("A-MV","j",-0.012),("A18","j",-0.020),("A17","c",-0.027)]
scen = [s for s in scen if s[0] != "A20*"]          # 빈attacks 제외(실공격 18)
scen.sort(key=lambda x: x[2])
labels = [s[0] for s in scen]
deltas = [s[2] for s in scen]
cls    = [s[1] for s in scen]
cmap   = {"j": TEAL, "c": AMBER}
barcol = [cmap[c] for c in cls]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios":[1, 1.35]})
fig.patch.set_facecolor("white")

# ---- 패널 A: 3개 시스템 점수 ----
x = range(len(systems))
bars = ax1.bar(x, scores, width=0.62, color=colors1, zorder=3)
for i, v in enumerate(scores):
    ax1.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
             fontsize=13, fontweight="bold", color=INK)
ax1.annotate("", xy=(2, 0.904), xytext=(0, 0.599),
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6,
                             connectionstyle="arc3,rad=-0.25"))
ax1.text(1.0, 0.74, "+0.305\n회복", ha="center", va="center", fontsize=11,
         color=MUTED, fontweight="bold")
ax1.axhline(0.906, color=SLATE, lw=1, ls="--", alpha=0.6, zorder=1)
ax1.set_xticks(list(x)); ax1.set_xticklabels(systems, fontsize=10.5, color=INK)
ax1.set_ylim(0, 1.02); ax1.set_ylabel("방어 점수", fontsize=11, color=MUTED)
ax1.set_title("① 방어점수 — RAG 구조개선 효과", fontsize=12.5, fontweight="bold",
              color=INK, pad=12)
ax1.grid(axis="y", color=GRID, lw=0.9, zorder=0)
for s in ["top","right"]: ax1.spines[s].set_visible(False)
for s in ["left","bottom"]: ax1.spines[s].set_color(GRID)
ax1.tick_params(colors=MUTED)

# ---- 패널 B: 시나리오별 Δ (개선RAG − HVT) ----
y = range(len(labels))
ax2.barh(y, deltas, color=barcol, height=0.66, zorder=3)
ax2.axvline(0, color=MUTED, lw=1.2, zorder=4)
for i, v in enumerate(deltas):
    ax2.text(v + (0.0016 if v >= 0 else -0.0016), i, f"{v:+.3f}",
             va="center", ha="left" if v >= 0 else "right",
             fontsize=8.3, color=MUTED)
ax2.set_yticks(list(y)); ax2.set_yticklabels(labels, fontsize=8.5, color=INK)
ax2.set_xlabel("Δ 방어점수  (개선 RAG - HVT)", fontsize=10.5, color=MUTED)
ax2.set_xlim(-0.05, 0.06)
ax2.set_title("② 시나리오별 승패 — 공격유형이 가른다", fontsize=12.5,
              fontweight="bold", color=INK, pad=12)
ax2.grid(axis="x", color=GRID, lw=0.9, zorder=0)
for s in ["top","right","left"]: ax2.spines[s].set_visible(False)
ax2.spines["bottom"].set_color(GRID); ax2.tick_params(colors=MUTED, length=0)
# 승패 주석
ax2.text(0.052, len(labels)-1.5, "재밍 = RAG 우세\n(FP 재장악 회피)", fontsize=9,
         color=TEAL, ha="right", va="top", fontweight="bold")
ax2.text(-0.048, 4.6, "감염 = HVT 우세\n(정밀 타겟팅)", fontsize=9,
         color=AMBER, ha="left", va="center", fontweight="bold")
# 범례
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(color=TEAL, label="재밍 class"),
                    Patch(color=AMBER, label="감염 class")],
           loc="lower right", frameon=False, fontsize=9)

fig.suptitle("HVT + RAG 최종 실험결과  ·  18 실공격 시나리오 · 5시드 · 실전탐지(r0.75/fp0.1)",
             fontsize=13.5, fontweight="bold", color=INK, y=1.005)
fig.text(0.5, -0.03, "개선 RAG v2 평균 0.904  (HVT 0.906 동률) · naive 0.599 대비 +0.305  ·  "
         "RAG = attack_class 라우팅 + NIST 봉쇄절차",
         ha="center", fontsize=9.5, color=MUTED)
plt.tight_layout()
plt.savefig("hvt_rag_results.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved hvt_rag_results.png")
