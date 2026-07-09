"""모델: evo_k6   (evo(EvoBlue) + k=6 최적화 θ)
레지스트리 정의: EvoBlue(n, theta_k6)
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
    def __init__(self, n): self.n = n


# 이 변형은 별도 클래스가 아니라 아래 base 클래스의 파라미터판입니다.
# 구조 = base 클래스 그대로, 등록만 다음과 같이:  EvoBlue(n, theta_k6)

# ── 공용 헬퍼 ──
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

class EvoBlue(BlueBrainBase):
    """임계값 벡터 θ로 파라미터화된 반사 정책 — 오프라인 최적화로 성능 상한 추정.

    θ (8차원)로 per-drone 행동과 예산 하 우선순위를 정한다. run_evo.py가 CMA-ES/랜덤서치로
    θ를 최적화한 뒤 고정 배포(결정론). "graph/whittle를 넘으면 병목은 아키텍처가 아니라
    파라미터"라는 상한 추정.

    θ = [retake_r, block_r, snr_gate, hub_frac, prio_spread_w, prio_dist_w,
         prio_selfheal, conserve_r]  (모두 0~1 정규화, 내부에서 스케일)
    """
    DIM = 8
    DEFAULT = [0.6, 0.9, 0.3, 0.2, 1.0, 0.5, 0.5, 0.5]

    def __init__(self, n: int, theta=None):
        super().__init__(n)
        self.theta = list(theta) if theta is not None else list(self.DEFAULT)
        self._prio = {}

    def _p(self):
        t = self.theta; R = 60.0
        return dict(retake_r=t[0]*R, block_r=t[1]*R, snr_gate=t[2]*20.0,
                    hub_frac=t[3], spread_w=t[4], dist_w=t[5],
                    selfheal=t[6], conserve_r=t[7]*R)

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos"); snr = ctx.get("snr")
        p = self._p()
        nbr = _proximity_graph(dict(ctx, n=self.n))
        # 위협원 = 감염 + 강재밍
        threats = set(comp)
        if snr is not None:
            for i in range(self.n):
                if snr[i] <= 6:
                    threats.add(i)
        # 감염원 전파 잠재력
        spread_pot = {c: len(nbr.get(c, set()) - comp) for c in comp}
        self._prio = {}
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                self._prio[d] = p["selfheal"] * 2.0     # 자가치유 우선순위(튜너블)
                continue
            if not threats or pos is None or d >= len(pos):
                aids.append(1); continue
            # 최근접 위협 + 거리
            dmin, cnear = 1e9, None
            for j in threats:
                if j < len(pos):
                    dd = _dist(pos, d, j)
                    if dd < dmin:
                        dmin, cnear = dd, j
            if dmin <= p["retake_r"]:
                aids.append(4)
                sp = spread_pot.get(cnear, 0)
                self._prio[d] = p["spread_w"]*sp - p["dist_w"]*(dmin/60.0)
            elif dmin <= p["block_r"]:
                aids.append(6)
                self._prio[d] = p["spread_w"]*spread_pot.get(cnear, 0)*0.5
            elif dmin <= p["conserve_r"]:
                aids.append(2)                          # Analyse (약대응)
            else:
                aids.append(1)
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
