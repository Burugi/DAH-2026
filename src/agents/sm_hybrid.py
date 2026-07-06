"""[구조 파악용] 모델: hybrid   (rule+rl+llm 융합 코어)
레지스트리 정의: HybridCoreBlue(n)
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

class HybridCoreBlue(BlueBrainBase):
    """rule 우선순위 + rl 가치추정 + llm 위협추론을 단일 코어로 융합 (ideation C3).

      · rule  : 자기 감염 → RemoveSessions (고정 우선순위)
      · llm   : 감염 비율 tier로 공격성 추론
      · rl    : 최근 보상 EMA(value)가 나쁘면 더 공격적으로 전환
    healthy 드론은 (위협 근접 or 고tier or 저value)면 능동 탈환, 아니면 분석(Analyse).
    """

    def __init__(self, n: int):
        super().__init__(n)
        self._val = 0.0        # reward EMA (rl-ish value)

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        f = len(comp) / max(1, self.n)
        tier = 0 if f == 0 else 1 if f < 0.15 else 2 if f < 0.4 else 3
        threats = _threat_set(ctx)
        radius = ctx.get("max_link", 40)
        aggressive = tier >= 2 or self._val < -8.0
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            if not comp:
                aids.append(1)
                continue
            if aggressive or _near_threat(d, ctx, threats, radius):
                aids.append(4)
            else:
                aids.append(2)      # Analyse (약한 대응)
        return aids

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self._val = 0.7 * self._val + 0.3 * reward

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)
