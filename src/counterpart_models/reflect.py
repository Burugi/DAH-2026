"""모델: reflect   (O-P-E-R: 행동 K회 후 반성)
레지스트리 정의: ReflectBlue(n, reflect_every=8)
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
    def __init__(self, n): self.n = n


class ReflectBlue(BlueBrainBase):
    """Act each step; Reflect every K steps to revise strategy.

    Reflexion-style loop:
      [Act × K]  →  Reflect (review outcomes, update strategy)
               →  [Act × K]  →  Reflect  →  …

    Strategy is stored as a text tag that selects a default action and
    influences the LLM prompt when the key is set.

    Observe  : current state
    Act      : follow current strategy
    Reflect  : (every K steps) review history → revise strategy
    """

    _STRATEGY_MAP = {
        "aggressive_retake": 4,   # RetakeSuspicious
        "block_and_isolate":  6,  # BlockSuspicious
        "remove_sessions":    3,  # RemoveSessions
        "monitor":            1,  # Monitor
        "analyse":            2,  # Analyse
    }

    def __init__(self, n: int, reflect_every: int = 8):
        super().__init__(n)
        self.reflect_every = reflect_every
        self.strategy: str = "monitor"
        self.strategy_desc: str = "Stay passive and monitor until threats appear."
        self.n_reflections: int = 0

    # ── Act ──────────────────────────────────────────────────────────────────
    def _stub_act(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        base = self._STRATEGY_MAP.get(self.strategy, 1)
        # Safety override: never monitor when drones are compromised
        if comp and base == 1:
            return 4
        return base

    def _act_prompt(self, ctx: dict) -> str:
        return (
            f"Current strategy: {self.strategy_desc}\n"
            f"State: {self._state_str(ctx)}\n\n"
            f"Pick the action id that best executes this strategy:\n{_MENU}"
        )

    # ── Reflect ───────────────────────────────────────────────────────────────
    def _reflect_prompt(self) -> str:
        recent = self.memory[-self.reflect_every:]
        rewards = [r["reward"] for r in recent]
        comp    = [r["compromised"] for r in recent]
        acts    = [r["action"] for r in recent]
        return (
            f"Review the last {len(recent)} steps of drone-swarm defense.\n\n"
            f"Current strategy : {self.strategy_desc}\n"
            f"Compromised trend: {comp}\n"
            f"Rewards          : {[round(r, 2) for r in rewards]}\n"
            f"Actions taken    : {acts}\n"
            f"Mean reward      : {round(sum(rewards)/max(1,len(rewards)), 3)}\n\n"
            f"Write ONE sentence describing the best strategy going forward.\n"
            f"Then output: STRATEGY=<tag>  where tag ∈ "
            f"[aggressive_retake | block_and_isolate | remove_sessions | monitor | analyse]"
        )

    def _reflect(self) -> None:
        if llm.available():
            try:
                client = llm._get_client()
                msg = client.messages.create(
                    model=llm.MODEL, max_tokens=80, temperature=0.0,
                    system="Tactical advisor. One sentence + STRATEGY=<tag>.",
                    messages=[{"role": "user",
                               "content": self._reflect_prompt()}])
                text = "".join(b.text for b in msg.content
                               if getattr(b, "type", "") == "text")
                # Extract human-readable description (first non-empty line)
                for line in text.strip().splitlines():
                    line = line.strip()
                    if line and not line.startswith("STRATEGY"):
                        self.strategy_desc = line
                        break
                # Extract strategy tag
                m = re.search(r"STRATEGY\s*=\s*(\w+)", text)
                if m and m.group(1) in self._STRATEGY_MAP:
                    self.strategy = m.group(1)
                self.n_reflections += 1
                return
            except Exception:
                pass

        # Offline stub: derive strategy from recent compromise trend
        recent = self.memory[-self.reflect_every:]
        avg_comp = sum(r["compromised"] for r in recent) / max(1, len(recent))
        last_comp = recent[-1]["compromised"] if recent else 0
        frac = last_comp / max(1, self.n)
        if frac > 0.50:
            self.strategy = "remove_sessions"
        elif avg_comp / self.n > 0.25:
            self.strategy = "aggressive_retake"
        elif avg_comp / self.n > 0.10:
            self.strategy = "block_and_isolate"
        else:
            self.strategy = "monitor"
        self.n_reflections += 1

    def step_decide(self, ctx: dict) -> int:
        # Reflect before deciding if on a reflection boundary
        if self.t > 0 and self.t % self.reflect_every == 0:
            self._reflect()

        return self._llm_id(
            self._act_prompt(ctx),
            "Defense agent. Reply with only the integer action id.",
            lambda: self._stub_act(ctx))

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self.memory[-1]["strategy"] = self.strategy
