"""모델: ooda   (O-P-E-R: 관찰-판단-결정-행동)
레지스트리 정의: OODABlue(n, window=5)
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
    def __init__(self, n): self.n = n


class OODABlue(BlueBrainBase):
    """Observe → Orient → Decide → Act (military OODA loop) each step.

    Observe : raw metrics — compromised count, delta, spreading trend
    Orient  : synthesize into threat level (none/low/medium/high/critical)
              and adversary model (spreading / persistent / both / none)
    Decide  : map threat × adversary → best action; LLM can override
    Act     : return the chosen id

    Loop: O → O → D → A  (every step)
    """

    # (threat_level, adversary_mode) → default action id
    _POLICY: dict[tuple[str, str], int] = {
        ("none",     "none"):       1,   # Monitor
        ("low",      "none"):       2,   # Analyse
        ("low",      "persistent"): 3,   # RemoveSessions
        ("medium",   "none"):       4,   # RetakeSuspicious
        ("medium",   "spreading"):  4,   # RetakeSuspicious
        ("medium",   "persistent"): 3,   # RemoveSessions
        ("medium",   "both"):       4,   # RetakeSuspicious
        ("high",     "none"):       4,   # RetakeSuspicious
        ("high",     "spreading"):  6,   # BlockSuspicious (stop the spread)
        ("high",     "persistent"): 3,   # RemoveSessions
        ("high",     "both"):       6,   # BlockSuspicious
        ("critical", "none"):       4,
        ("critical", "spreading"):  6,
        ("critical", "persistent"): 4,
        ("critical", "both"):       6,
    }

    def __init__(self, n: int, window: int = 5):
        super().__init__(n)
        self.window = window
        self._threat: str = "none"
        self._mode: str = "none"
        self._comp_hist: list[int] = []

    # ── Observe ───────────────────────────────────────────────────────────────
    def _observe(self, ctx: dict) -> dict:
        comp = ctx["compromised"]
        prev = self._comp_hist[-1] if self._comp_hist else 0
        trend = self._comp_hist[-self.window:]
        return {
            "count":     len(comp),
            "delta":     len(comp) - prev,
            "trend":     trend,
            "spreading": len(comp) > prev,
        }

    # ── Orient ───────────────────────────────────────────────────────────────
    def _orient(self, obs: dict) -> tuple[str, str]:
        frac   = obs["count"] / max(1, self.n)
        delta  = obs["delta"]
        trend  = obs["trend"]

        # Threat level
        if frac == 0:
            level = "none"
        elif frac < 0.10:
            level = "low"
        elif frac < 0.30:
            level = "medium"
        elif frac < 0.55:
            level = "high"
        else:
            level = "critical"

        # Adversary mode: is it spreading? persisting?
        spreading  = delta > 0 or (len(trend) >= 2 and trend[-1] > trend[-2])
        persistent = (len(trend) >= 3
                      and all(t >= trend[0] for t in trend[-3:])
                      and delta == 0)
        if spreading and persistent:
            mode = "both"
        elif spreading:
            mode = "spreading"
        elif persistent and frac > 0:
            mode = "persistent"
        else:
            mode = "none"

        return level, mode

    # ── Decide ────────────────────────────────────────────────────────────────
    def _stub_decide(self, level: str, mode: str) -> int:
        return self._POLICY.get((level, mode),
               self._POLICY.get((level, "none"), 1))

    def _decide_prompt(self, obs: dict, level: str, mode: str) -> str:
        return (
            f"OODA Defense — drone swarm.\n"
            f"[Observe] compromised={obs['count']}/{self.n}  "
            f"delta={obs['delta']:+d}  trend={obs['trend']}\n"
            f"[Orient]  threat={level}  adversary_mode={mode}\n\n"
            f"[Decide]  Which action id best counters this situation?\n"
            f"{_MENU}"
        )

    # ── Act (entry point) ─────────────────────────────────────────────────────
    def step_decide(self, ctx: dict) -> int:
        obs = self._observe(ctx)
        self._comp_hist.append(obs["count"])
        level, mode = self._orient(obs)
        self._threat, self._mode = level, mode

        return self._llm_id(
            self._decide_prompt(obs, level, mode),
            "Defense agent. Reply with only the integer action id.",
            lambda: self._stub_decide(level, mode))

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        r = self.memory[-1]
        r["threat"] = self._threat
        r["mode"]   = self._mode
