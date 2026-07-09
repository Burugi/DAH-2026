"""모델: plan   (O-P-E-R: 계획-실행-재계획)
레지스트리 정의: PlannerBlue(n, replan_threshold=0.45)
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
    def __init__(self, n): self.n = n


class PlannerBlue(BlueBrainBase):
    """Plan at episode start → Execute → Re-plan on large deviations.

    The plan is a list of (until_step, action_id) phases executed in order.
    Re-planning fires when compromise fraction exceeds replan_threshold
    (with a cooldown to avoid thrashing).

    Plan     : generate at step 0 (LLM or stub)
    Execute  : follow current phase
    Re-plan  : trigger if situation diverges significantly
    """

    _DEFAULT_PLAN = [(8, 1), (20, 4), (35, 3), (999, 4)]
    # Monitor until 8 → RetakeSuspicious until 20 → RemoveSessions until 35 → Retake

    def __init__(self, n: int, replan_threshold: float = 0.45,
                 replan_cooldown: int = 6):
        super().__init__(n)
        self.replan_threshold = replan_threshold
        self.replan_cooldown = replan_cooldown
        self._plan: list[tuple[int, int]] = []
        self._last_replan_t: int = -999
        self._n_replans: int = 0

    # ── Plan ─────────────────────────────────────────────────────────────────
    def _plan_prompt(self, ctx: dict) -> str:
        return (
            f"Plan a defense strategy for a ~40-step drone-swarm episode.\n"
            f"Initial state: {self._state_str(ctx)}\n\n"
            f"Output a JSON list of [until_step, action_id] phases.\n"
            f"Example: [[8,1],[20,4],[35,3],[40,4]]\n"
            f"= Monitor(1) until step 8 → RetakeSuspicious(4) until 20 "
            f"→ RemoveSessions(3) until 35 → RetakeSuspicious(4) to end.\n\n"
            f"Available actions (ids 0-8):\n{_MENU}\n\n"
            f"Output ONLY the JSON list."
        )

    def _make_plan(self, ctx: dict) -> list[tuple[int, int]]:
        if llm.available():
            try:
                client = llm._get_client()
                msg = client.messages.create(
                    model=llm.MODEL, max_tokens=60, temperature=0.0,
                    system="Output only valid JSON. No explanation.",
                    messages=[{"role": "user",
                               "content": self._plan_prompt(ctx)}])
                text = "".join(b.text for b in msg.content
                               if getattr(b, "type", "") == "text")
                m = re.search(r"\[.*?\]", text, re.DOTALL)
                if m:
                    raw = json.loads(m.group(0))
                    plan = [(int(p[0]), int(p[1])) for p in raw
                            if isinstance(p, (list, tuple)) and len(p) == 2
                            and 0 <= int(p[1]) < BLUE_DECISION_N]
                    if plan:
                        return plan
            except Exception:
                pass

        # Stub: adaptive plan based on initial threat level
        comp_frac = len(ctx["compromised"]) / max(1, self.n)
        if comp_frac > 0.20:                            # already under attack
            return [(5, 3), (20, 4), (35, 6), (999, 4)]
        return list(self._DEFAULT_PLAN)

    # ── Execute ───────────────────────────────────────────────────────────────
    def _phase_aid(self) -> int:
        for until, aid in self._plan:
            if self.t <= until:
                return aid
        return 4    # default: RetakeSuspicious if plan exhausted

    def _should_replan(self, ctx: dict) -> bool:
        if self.t - self._last_replan_t < self.replan_cooldown:
            return False
        return len(ctx["compromised"]) / max(1, self.n) > self.replan_threshold

    def step_decide(self, ctx: dict) -> int:
        if self.t == 0:
            self._plan = self._make_plan(ctx)
            self._last_replan_t = 0

        if self.t > 0 and self._should_replan(ctx):
            self._plan = self._make_plan(ctx)
            self._last_replan_t = self.t
            self._n_replans += 1

        return self._phase_aid()

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self.memory[-1]["n_replans"] = self._n_replans
