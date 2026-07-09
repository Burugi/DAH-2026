"""모델: mincut   (그래프 최소절단 방화벽 할당)
레지스트리 정의: MinCutBlue(n)
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
    def __init__(self, n): self.n = n


# 공용 헬퍼
def _dist(pos, i, j):
    return math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])

def _threat_set(ctx: dict) -> set:
    """위협원 = 감염 드론 + 강재밍(SNR<=6) 드론."""
    comp = set(ctx["compromised"])
    snr = ctx.get("snr")
    if snr is not None:
        for i in range(len(snr)):
            if snr[i] <= 6:
                comp.add(i)
    return comp

def _near_threat(d: int, ctx: dict, threats: set, radius: float) -> bool:
    pos = ctx.get("pos")
    if pos is None or d >= len(pos) or not threats:
        return bool(threats)
    dmin = min((_dist(pos, d, j) for j in threats if j < len(pos)), default=1e9)
    return dmin <= radius

def _proximity_graph(ctx):
    """근접 인접 리스트: nbr[i] = i의 max_link 내 이웃 집합 (연결된 드론만)."""
    pos = ctx.get("pos"); link = ctx.get("link_up"); R = ctx.get("max_link", 40)
    n = ctx["n"] if "n" in ctx else (len(pos) if pos is not None else 0)
    nbr = {i: set() for i in range(n)}
    if pos is None:
        return nbr
    up = lambda i: (link is None) or bool(link[i])
    for i in range(min(n, len(pos))):
        if not up(i):
            continue
        for j in range(i + 1, min(n, len(pos))):
            if up(j) and _dist(pos, i, j) <= R:
                nbr[i].add(j); nbr[j].add(i)
    return nbr

class MinCutBlue(BlueBrainBase):
    """그래프 최소절단 방화벽 (근사).

    감염집합과 건강집합 사이의 경계(프론티어)를 위상적으로 끊는다. 프론티어 = 감염
    드론과 인접한 건강 드론. 프론티어 건강 드론은 감염 이웃 수가 많을수록 우선 탈환/차단
    (절단 효과 큼). 그 외 건강 드론은 벽 뒤에서 보존(Monitor).
    """
    def __init__(self, n: int):
        super().__init__(n)
        self._prio = {}

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]
        nbr = _proximity_graph(dict(ctx, n=self.n))
        self._prio = {}
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            inf_nb = nbr.get(d, set()) & comp
            if inf_nb:                                   # 프론티어 건강 드론
                self._prio[d] = float(len(inf_nb))       # 절단 효과 = 감염 이웃 수
                # 감염 이웃이 많으면 탈환(4), 하나면 차단(6)으로 간선 제거
                aids.append(4 if len(inf_nb) >= 2 else 6)
            else:
                aids.append(1)                           # 벽 뒤 → 보존
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
