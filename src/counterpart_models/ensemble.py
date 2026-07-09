"""모델: ensemble   (전문가 혼합(MoE) 다수결)
레지스트리 정의: EnsembleMoEBlue(n)
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

class EnsembleMoEBlue(BlueBrainBase):
    """전문가 혼합(MoE) — graph·predictive·rule 전문가의 드론별 제안을 봉쇄우선 결합.

    세 전문가가 각자 per-drone action을 제안하면, 드론마다 '봉쇄 우선순위'가 가장 높은
    행동을 채택(게이팅). 서로 다른 전문성을 한 팀 행동으로 합성한다.
    """

    # 동점 시 선호 순위(앞일수록 우선) — graph 전문가 성향 반영
    _PRIORITY = [4, 3, 9, 6, 5, 7, 2, 1, 0]

    def __init__(self, n: int):
        super().__init__(n)
        self._graph = GraphCentralityBlue(n)
        self._pred = PredictiveBlue(n)

    def _rule_aids(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        return [3 if int(a.split("_")[-1]) in comp else (4 if comp else 1)
                for a in agents]

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        votes = [self._graph.team_decide(ctx, agents),
                 self._pred.team_decide(ctx, agents),
                 self._rule_aids(ctx, agents)]
        rank = lambda x: self._PRIORITY.index(x) if x in self._PRIORITY else 99
        out = []
        for i in range(len(agents)):
            cand = [v[i] for v in votes]
            # 다수결; 동점이면 선호 순위로 결정
            counts = {}
            for c in cand:
                counts[c] = counts.get(c, 0) + 1
            top = max(counts.values())
            tied = [c for c in counts if counts[c] == top]
            out.append(min(tied, key=rank))
        return out

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)
