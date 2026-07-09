"""모델: rule   (휴리스틱 FSM 베이스라인)
레지스트리 정의: blue_decide(btype='rule')
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
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

