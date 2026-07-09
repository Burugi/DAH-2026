"""모델: stigmergy   (페로몬 스티그머지(분산))
레지스트리 정의: StigmergyBlue(n)
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

class StigmergyBlue(BlueBrainBase):
    """페로몬 격자 스티그머지 — 완전 분산 창발 분업 (개미군집형).

    격자(grid×grid 축소)에 위험 페로몬 필드를 유지: 감염 드론 위치에 침적, 매 스텝
    확산(이웃 평균)·증발(×decay). 각 드론은 자기 위치의 페로몬 농도만 보고 로컬 규칙으로
    행동한다(중앙 조정 없음). 중앙집중(graph) vs 분산 조정비용 비교 소재.
    """
    def __init__(self, n: int, cells: int = 20, decay: float = 0.85,
                 deposit: float = 1.0, hi: float = 0.5, mid: float = 0.15):
        super().__init__(n)
        self.cells = cells; self.decay = decay; self.deposit = deposit
        self.hi = hi; self.mid = mid
        self.field = None

    def _cell(self, ctx, d):
        pos = ctx.get("pos"); grid = 100.0
        if pos is None or d >= len(pos):
            return 0, 0
        cx = min(self.cells-1, int(pos[d][0]/grid*self.cells))
        cy = min(self.cells-1, int(pos[d][1]/grid*self.cells))
        return cx, cy

    def _update_field(self, ctx):
        import numpy as np
        if self.field is None:
            self.field = np.zeros((self.cells, self.cells))
        self.field *= self.decay                                  # 증발
        for c in ctx["compromised"]:                              # 침적
            cx, cy = self._cell(ctx, c)
            self.field[cx, cy] += self.deposit
        # 확산(간이 박스블러)
        f = self.field
        self.field = (f + np.roll(f,1,0) + np.roll(f,-1,0)
                      + np.roll(f,1,1) + np.roll(f,-1,1)) / 5.0

    def team_decide(self, ctx, agents):
        self._update_field(ctx)
        comp = ctx["compromised"]
        mx = float(self.field.max()) or 1.0
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            cx, cy = self._cell(ctx, d)
            conc = float(self.field[cx, cy])/mx
            if conc >= self.hi:
                aids.append(4)                   # 고농도 → 능동 탈환
            elif conc >= self.mid:
                aids.append(6)                   # 중농도 → 차단
            else:
                aids.append(1)                   # 저농도 → 감시(보존)
        return aids

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
