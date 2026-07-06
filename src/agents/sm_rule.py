"""[구조 파악용] 모델: rule   (휴리스틱 FSM 베이스라인)
레지스트리 정의: blue_decide(btype='rule')
※ 팀원 구조 파악 전용 — 실제 실행/임포트는 원본(src/agents/brains.py)을 사용.
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


# rule 베이스라인 — 클래스가 아니라 brains.blue_decide()의 per-agent 분기.
# 드론별로 독립 호출되는 간단한 FSM(휴리스틱).
def blue_decide_rule(agent, ctx):
    """rule blue: 한 드론의 이번 스텝 행동 id 반환 (0-9)."""
    own = int(agent.split("_")[-1])
    comp = ctx["compromised"]                 # 감염 드론 집합
    if own in comp:
        return 3                              # 자기 감염 → RemoveSessions(세션 제거)
    if comp:
        return 4                              # 감염 보이면 → RetakeSuspicious(탈환)
    return 1                                  # 없으면 → Monitor(감시)

