# -*- coding: utf-8 -*-
"""시나리오 동작 — 웜 감염 확산 vs 방어 (스텝별 감염률, 무방어/HVT/HVT+RAG)."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ["Malgun Gothic","맑은 고딕","AppleGothic","NanumGothic"]:
    if any(f.lower() in x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False

INK,MUTED,GRID="#1f2733","#5b6875","#e7ebf0"
NONE,HVT,RAG="#cc6a5c","#8492a3","#149e8c"
D=json.load(open("rollout_curves.json"))
titles={"A17":"A17 — 스웜 C2 탈취 (웜 확산)","A9":"A9 — GPS+웜 혼합","A14":"A14 — 재밍+웜 혼합"}
order=["A17","A9","A14"]

fig,axes=plt.subplots(1,3,figsize=(13.5,4.4),sharey=True); fig.patch.set_facecolor("white")
for ax,sid in zip(axes,order):
    d=D[sid]; steps=range(len(d["none"]))
    ax.plot(steps,[v*100 for v in d["none"]],color=NONE,lw=2.4,label="무방어",zorder=3)
    ax.plot(steps,[v*100 for v in d["hvt"]], color=HVT, lw=2.2,label="HVT",zorder=4)
    ax.plot(steps,[v*100 for v in d["rag"]], color=RAG, lw=2.2,ls="--",label="HVT+RAG",zorder=5)
    ax.fill_between(steps,[v*100 for v in d["none"]],[v*100 for v in d["hvt"]],
                    color=NONE,alpha=0.07,zorder=1)
    ax.set_title(titles[sid],fontsize=11.5,fontweight="bold",color=INK,pad=8)
    ax.set_xlabel("스텝",fontsize=10,color=MUTED)
    ax.grid(color=GRID,lw=0.8,zorder=0)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.set_ylim(0,90)
    # 최대 감염 주석
    nmax=max(d["none"])*100; hmax=max(d["hvt"])*100
    ax.annotate(f"{nmax:.0f}%",xy=(len(steps)-1,nmax),xytext=(-4,4),
                textcoords="offset points",fontsize=9,color=NONE,fontweight="bold",ha="right")
    ax.annotate(f"{hmax:.0f}%",xy=(len(steps)-1,hmax),xytext=(-4,6),
                textcoords="offset points",fontsize=9,color=HVT,fontweight="bold",ha="right")
axes[0].set_ylabel("감염 드론 비율 (%)",fontsize=10.5,color=MUTED)
axes[0].legend(loc="center left",frameon=False,fontsize=10)
fig.suptitle("시나리오 동작 — 웜 감염 확산 vs 방어 억제  (스텝별 감염률, 5시드 평균)",
             fontsize=13.5,fontweight="bold",color=INK,y=1.02)
fig.text(0.5,-0.04,"무방어는 감염이 최대 ~78%까지 확산 · HVT/HVT+RAG는 belief-검증-재장악으로 ~17-19%에서 억제 "
         "(초기 seed 감염만 남고 확산 차단)",ha="center",fontsize=9,color=MUTED)
plt.tight_layout()
plt.savefig("scenario_dynamics.png",dpi=170,bbox_inches="tight",facecolor="white")
print("saved scenario_dynamics.png")
