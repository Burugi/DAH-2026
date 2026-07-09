"""모델: debate   (다중 페르소나 토론)
레지스트리 정의: DebateBlue(n)
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

class DebateBlue(BlueBrainBase):
    """다중 페르소나 토론/자기일관성 — Hawk(최대 봉쇄) vs Dove(보존) 제안 → 심판 채택.

    두 페르소나가 드론별 행동을 제안하고, 확산 여부·위협 tier로 심판이 드론별로 채택한다.
    확산 중이거나 고tier면 Hawk(능동 탈환), 아니면 Dove(위협 근접 시만 탈환, 그 외 보존).
    """

    def __init__(self, n: int):
        super().__init__(n)

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        prev = self.memory[-1]["compromised"] if self.memory else 0
        spreading = len(comp) > prev
        f = len(comp) / max(1, self.n)
        threats = _threat_set(ctx)
        radius = ctx.get("max_link", 40)
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            if not comp:
                aids.append(1)
                continue
            hawk = 4
            dove = 4 if _near_threat(d, ctx, threats, radius) else 1
            aids.append(hawk if (spreading or f >= 0.4) else dove)
        return aids

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)
