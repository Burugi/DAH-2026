"""Shared base class for all stateful blue brain architectures.

Split into its own module to avoid circular imports between multiagent.py
(which registers concrete subclasses) and hierarchical.py (which also subclasses).
"""
from __future__ import annotations
import re

from agents import llm
from agents.actions import BLUE_DECISION_N, BLUE_CATALOG

_ACT_NAMES = [c[0] for c in BLUE_CATALOG[:BLUE_DECISION_N]]


class BlueBrainBase:
    """Stateful blue commander — one instance per rollout episode."""

    def __init__(self, n: int):
        self.n = n                          # total drones in fleet
        self.t = 0                          # current step (advanced by step_end)
        self.memory: list[dict] = []        # per-step records

    # ── public interface ───────────────────────────────────────────────────

    def step_decide(self, ctx: dict) -> int:
        """Return ONE blue catalog id for the whole team this step."""
        raise NotImplementedError

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        """Record outcome and advance the step counter."""
        self.memory.append({
            "t":           self.t,
            "compromised": len(ctx["compromised"]),
            "reward":      round(reward, 3),
            "aid":         aid,
            "action":      _ACT_NAMES[aid] if aid < len(_ACT_NAMES) else str(aid),
        })
        self.t += 1

    def episode_end(self) -> None:
        pass

    # ── shared helpers ─────────────────────────────────────────────────────

    def _state_str(self, ctx: dict) -> str:
        comp = ctx["compromised"]
        return (f"step={self.t}  compromised={len(comp)}/{self.n}  "
                f"ids={sorted(comp) or 'none'}")

    def _history_str(self, k: int = 5) -> str:
        rows = self.memory[-k:]
        if not rows:
            return ""
        lines = [f"  t={r['t']}: {r['action']:22s} comp={r['compromised']}/{self.n}"
                 f"  rew={r['reward']:+.3f}"
                 for r in rows]
        return "Recent history:\n" + "\n".join(lines)

    def _rule_fallback(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        if not comp:
            return 1    # Monitor
        if len(comp) / self.n > 0.4:
            return 3    # RemoveSessions (urgent)
        return 4        # RetakeSuspicious

    def _llm_id(self, prompt: str, system: str, stub_fn) -> int:
        """Call Claude for a valid catalog id; fall back to stub on any error."""
        if not llm.available():
            return stub_fn()
        try:
            client = llm._get_client()
            msg = client.messages.create(
                model=llm.MODEL, max_tokens=16, temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text")
            nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
            for num in nums:
                if 0 <= num < BLUE_DECISION_N:
                    return num
        except Exception:
            pass
        return stub_fn()
