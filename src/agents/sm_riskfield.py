"""[구조 파악용] 모델: riskfield   (베이지안 위험장 선제방어)
레지스트리 정의: RiskFieldBlue(n)
※ 팀원 구조 파악 전용 — 실제 실행/임포트는 원본(src/agents/experimental.py)을 사용.
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
# ─────────────── 참조표 (전 모델 공통) ───────────────
# 행동 id(team_decide/step_decide 반환): 0 Sleep · 1 Monitor(감시) · 2 Analyse(분석)
#   · 3 RemoveSessions(자가치유:자기세션제거) · 4 RetakeSuspicious(탈환:재이미징)
#   · 5 RetakeRandom · 6 BlockSuspicious(차단) · 7 AllowTraffic · 8 DeployDecoy · 9 Failsafe
# ctx 키(관측 입력): compromised(감염 드론 id 집합) · n(드론수) · pos(위치[x,y] 리스트)
#   · snr(신호세기, 낮을수록 재밍) · gps_err(GPS오차) · link_up(드론별 연결여부)
#   · max_link(연결 판정 반경) · ip_to_drone
# 예산 캡: 스텝당 능동봉쇄({3,4,5,6}) 수를 k로 상한, 초과분은 recovery_priority 낮은 순 Monitor 강등
# ────────────────────────────────────────────────────
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 구조 파악용 최소 스텁
    def __init__(self, n): self.n = n


# ── 공용 헬퍼(원본 experimental.py 모듈 레벨) ──
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

class RiskFieldBlue(BlueBrainBase):
    """베이지안 위험장 선제 방어 — 미래 감염을 사전 대응.

    기존 대부분이 compromised 라벨에 '반응'하는 데 반해, 각 healthy 드론의 다음 스텝
    감염확률을 감염원 근접·SNR 열화로 추정해 '고위험 미감염' 노드에 선제 조치한다.
      P(감염_d) ∝ Σ_{c∈compromised} exp(-dist(d,c)/σ)  (+ 낮은 SNR 가중)
    고위험 → DeployDecoy(8) 유인/차단, 중위험 → BlockSuspicious(6), 감염 → 탈환/제거.
    액션 8을 실제로 쓰는 드문 아키텍처(액션 커버리지 분석 소재).
    """
    def __init__(self, n: int, sigma: float = 25.0, hi: float = 0.6, mid: float = 0.3):
        super().__init__(n)
        self.sigma = sigma; self.hi = hi; self.mid = mid

    def _risk(self, ctx):
        comp = ctx["compromised"]; pos = ctx.get("pos"); snr = ctx.get("snr")
        risk = [0.0]*self.n
        if pos is None or not comp:
            return risk
        for d in range(min(self.n, len(pos))):
            if d in comp:
                continue
            r = sum(math.exp(-_dist(pos, d, c)/self.sigma) for c in comp if c < len(pos))
            if snr is not None and snr[d] <= 6:
                r *= 1.5                         # 재밍당한 노드는 더 취약
            risk[d] = r
        return risk

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]
        risk = self._risk(ctx)
        mx = max(risk) or 1.0
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(4 if len(comp)/self.n < 0.5 else 3)
                continue
            rn = risk[d]/mx
            if rn >= self.hi:
                aids.append(6)                   # 고위험 → 선제 차단
            elif rn >= self.mid:
                aids.append(8)                   # 중위험 → 디코이 배치
            elif comp:
                aids.append(4)                   # 저위험이나 위협 존재 → 탈환 가담
            else:
                aids.append(1)
        return aids

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
