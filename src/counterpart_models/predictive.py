"""모델: predictive   (SIR 확산예측 선제방어)
레지스트리 정의: PredictiveBlue(n, base_radius=25, block_top=2)
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

class PredictiveBlue(BlueBrainBase):
    """SIR형 확산 예측 기반 선제 방어.

    웜은 감염 드론에서 근접한 healthy 드론으로 전파된다(run._worm_step: proximity).
    따라서 '다음에 감염될 드론' = 감염원 반경 내 healthy 드론이다. 이 아키텍처는:

      · 감염원(spreader) 중 healthy 이웃이 많은 노드를 우선 차단(BlockSuspicious)해
        전파 front를 끊는다.
      · front에 인접한 healthy 드론은 guardian으로 감염 이웃을 능동 탈환(RetakeSuspicious).
      · front에서 먼 healthy 드론은 reserve로 보존(Monitor) — 공격면·행동낭비 축소.

    관측된 확산 속도(최근 Δcompromised)로 guardian 반경을 동적 조절한다:
    확산이 가속 중이면 반경을 넓혀 공격적으로, 안정적이면 좁혀 보존적으로.
    """

    def __init__(self, n: int, base_radius: float = 25.0, block_top: int = 2):
        super().__init__(n)
        self.base_radius = base_radius
        self.block_top = block_top          # 매 스텝 차단할 상위 spreader 수
        self._comp_hist: list[int] = []

    def _spread_rate(self) -> int:
        """최근 2스텝 감염 증가량(양수면 확산 중)."""
        h = self._comp_hist
        if len(h) < 2:
            return 0
        return h[-1] - h[-2]

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        pos = ctx.get("pos")
        max_link = ctx.get("max_link", 40)
        self._comp_hist.append(len(comp))

        # 확산 가속 시 반경 확대(1.4배), 안정 시 축소(0.7배)
        rate = self._spread_rate()
        radius = self.base_radius * (1.4 if rate > 0 else 0.7 if len(comp) else 1.0)

        # 감염원별 전파 잠재력 = 반경 내 healthy 이웃 수 → 상위 spreader 선정
        spreaders: set = set()
        if pos is not None and comp:
            potential = {}
            for c in comp:
                if c >= len(pos):
                    continue
                nb = sum(1 for d in range(self.n)
                         if d not in comp and d < len(pos) and _dist(pos, c, d) <= radius)
                potential[c] = nb
            spreaders = set(sorted(potential, key=lambda x: -potential[x])[:self.block_top])

        aids = []
        for agent in agents:
            d = int(agent.split("_")[-1])
            if d in comp:
                # 감염 노드: 전파력 큰 spreader면 차단, 아니면 세션 제거
                aids.append(6 if d in spreaders else 3)
                continue
            if not comp or pos is None or d >= len(pos):
                aids.append(1)                    # 위협 없음 → 감시
                continue
            dmin = min((_dist(pos, d, c) for c in comp if c < len(pos)), default=1e9)
            if dmin <= radius:
                aids.append(4)                    # front 인접 guardian → 능동 탈환
            else:
                aids.append(1)                    # reserve → 보존
        return aids

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        self._comp_hist.append(len(comp))
        if not comp:
            return 1
        return 6 if len(comp) / self.n < 0.5 else 3
