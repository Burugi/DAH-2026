"""모델: whittle   (휘틀/cμ 인덱스 스케줄러 할당)
레지스트리 정의: WhittleBlue(n)
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

class WhittleBlue(BlueBrainBase):
    """휘틀/cμ 인덱스 스케줄러 — 학습 없는 폐형 우선순위.

    감염 드론(대기 작업)의 인덱스 = 다음 스텝 기대 전염수(반경 내 건강 이웃 수) /
    기대 탈환비용(최근접 건강 드론 거리). 건강 드론(서버)을 인덱스 큰 작업부터 배정.
    """
    def __init__(self, n: int, selfheal: float = 0.0):
        super().__init__(n)
        self._prio = {}
        # selfheal>0: 감염 드론 자가치유(RemoveSessions)에 전파잠재력 기반 우선순위 부여.
        # 기본 0(원판)은 자가치유 우선순위=0이라 예산 하에서 굶김 → k=3 딥의 원인(가설).
        # EvoBlue가 k=6에서 자가치유 우선을 스스로 발견한 것을 whittle에 역이식(지식전이).
        self.selfheal = selfheal

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos")
        nbr = _proximity_graph(dict(ctx, n=self.n))
        healthy = [int(a.split("_")[-1]) for a in agents
                   if int(a.split("_")[-1]) not in comp]
        spread_pot = {c: len(nbr.get(c, set()) - comp) for c in comp}   # 감염원 전파잠재력
        # 작업 인덱스
        index = {}
        for c in comp:
            if pos is not None and c < len(pos) and healthy:
                cost = min((_dist(pos, c, d) for d in healthy if d < len(pos)), default=1e9)
            else:
                cost = 1.0
            index[c] = spread_pot[c] / (1.0 + cost)
        # 인덱스 큰 작업부터 최근접 건강 드론 배정
        assign = {}; self._prio = {}
        avail = set(healthy)
        for c in sorted(index, key=lambda x: -index[x]):
            if pos is None or c >= len(pos) or not avail:
                continue
            d = min(avail, key=lambda x: _dist(pos, x, c) if x < len(pos) else 1e9)
            assign[d] = True; self._prio[d] = index[c]; avail.discard(d)
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                if self.selfheal:
                    # 자가치유 우선순위 = 자신의 전파잠재력 × selfheal (탈환 인덱스와 동일 스케일)
                    self._prio[d] = self.selfheal * spread_pot.get(d, 0)
            elif assign.get(d):
                aids.append(4)
            else:
                aids.append(1)
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
