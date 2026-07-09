# -*- coding: utf-8 -*-
"""논문용 모델 성능 비교 (실전조건 채널①, recall0.75/fp0.1)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ["Malgun Gothic","맑은 고딕","AppleGothic","NanumGothic"]:
    if any(f.lower() in x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False

INK,MUTED,GRID="#1f2733","#5b6875","#e7ebf0"
OURS,OURS_HI,TEAM,NONE="#3aa89a","#0d6e62","#8492a3","#cc6a5c"

# (모델, 점수, 진영, 출처)  실전조건 채널① recall0.75/fp0.1
data=[("HVT+RAG (최종)",0.915,"ours_hi"),
      ("Coordinator",   0.918,"ours"),
      ("whittle SH",    0.823,"team"),
      ("Graph",         0.810,"team"),
      ("EVO",           0.783,"team"),
      ("hier_h3_tight", 0.735,"team"),
      ("Rule",          0.731,"team"),
      ("None (무방어)", 0.398,"none")]
data.sort(key=lambda x:x[1])                     # 아래→위 오름차순
labels=[d[0] for d in data]; vals=[d[1] for d in data]
cmap={"ours_hi":OURS_HI,"ours":OURS,"team":TEAM,"none":NONE}
cols=[cmap[d[2]] for d in data]

fig,ax=plt.subplots(figsize=(9.2,5.4)); fig.patch.set_facecolor("white")
y=range(len(labels))
bars=ax.barh(y,vals,height=0.66,color=cols,zorder=3)
for i,v in enumerate(vals):
    ax.text(v+0.008,i,f"{v:.3f}",va="center",ha="left",fontsize=11,
            fontweight="bold" if data[i][2] in("ours_hi","ours") else "normal",
            color=INK if data[i][2] in("ours_hi","ours") else MUTED)
ax.set_yticks(list(y)); ax.set_yticklabels(labels,fontsize=11,color=INK)
ax.set_xlim(0,1.0); ax.set_xlabel("방어 점수 (채널① = mean(1-점령, 1-AUC))  ·  ↑ 높을수록 우수",
                                   fontsize=10.5,color=MUTED)
ax.set_title("드론군집 방어 모델 성능 비교",fontsize=14,fontweight="bold",color=INK,pad=14)
ax.grid(axis="x",color=GRID,lw=0.9,zorder=0)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED,length=0)
# None 기준선
ax.axvline(0.398,color=NONE,ls=":",lw=1,alpha=0.5,zorder=1)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=OURS_HI,label="본 연구 (최종)"),
                   Patch(color=OURS,label="본 연구"),
                   Patch(color=TEAM,label="팀/기존 모델"),
                   Patch(color=NONE,label="무방어 기준")],
          loc="lower right",frameon=False,fontsize=9.5)
fig.text(0.5,-0.02,"실전조건: recall 0.75 · fp 0.1 · 미탐/오탐 동시 · 23 시나리오 · seeds 0-4  ·  "
         "HVT+RAG·None=직접측정(HVT 재현 0.892 ~ LB 0.906 정합), 그외=팀 벤치",
         ha="center",fontsize=8.6,color=MUTED)
plt.tight_layout()
plt.savefig("model_comparison.png",dpi=170,bbox_inches="tight",facecolor="white")
print("saved model_comparison.png")
