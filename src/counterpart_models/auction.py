"""모델: auction   (시장기반 경매(CBBA) 할당)
레지스트리 정의: AuctionBlue(n)
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

class AuctionBlue(BlueBrainBase):
    """시장기반 태스크 경매 (CBBA류).

    감염 드론 = 태스크, 건강 드론 = 입찰자. 입찰가 = f(거리, SNR). 그리디 순차경매로
    각 태스크를 최고 입찰 건강 드론에 1:1 배정. 낙찰자는 Retake(4), 유찰 태스크 인접
    건강 드론은 Block(6), 나머지 Monitor(1). 예산 하에서 낙찰 입찰가가 곧 우선순위.
    """
    def __init__(self, n: int):
        super().__init__(n)
        self._bid = {}      # drone -> 낙찰 입찰가 (recovery_priority용)

    def _auction(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos"); snr = ctx.get("snr")
        healthy = [int(a.split("_")[-1]) for a in agents
                   if int(a.split("_")[-1]) not in comp]
        tasks = list(comp)
        assign = {}                     # drone -> True(낙찰)
        self._bid = {}
        if pos is not None:
            avail = set(healthy)
            for task in tasks:
                if task >= len(pos):
                    continue
                best, bestbid = None, -1.0
                for d in avail:
                    if d >= len(pos):
                        continue
                    dd = _dist(pos, d, task)
                    bid = 1.0 / (1.0 + dd) * (1.0 + max(0.0, (snr[d] if snr is not None else 20) / 20.0))
                    if bid > bestbid:
                        best, bestbid = d, bid
                if best is not None:
                    assign[best] = True; self._bid[best] = bestbid; avail.discard(best)
        return assign

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos")
        assign = self._auction(ctx, agents)
        nbr = _proximity_graph(dict(ctx, n=self.n))
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
            elif assign.get(d):
                aids.append(4)                          # 낙찰 → 탈환
            elif nbr.get(d, set()) & comp:
                aids.append(6)                          # 유찰 태스크 인접 → 차단
            else:
                aids.append(1)
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._bid.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
