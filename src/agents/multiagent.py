"""Multi-agent blue brain architectures with Observe-Plan/React-Execute-Reflect loops.

Each architecture is a *stateful* object (one instance per rollout episode) that
produces ONE team-level action id per step.  That id is then applied to every live
drone via make_blue_index(), which handles drone-specific targeting internally.
This satisfies the "1 step = 1 representative action" contract.

Architecture  | Loop                            | LLM calls / step
──────────────┼─────────────────────────────────┼──────────────────
react         | Observe → Reason (CoT) → Act    | 1 per step
reflect       | Act × K  →  Reflect  →  Act × K | 1 every K steps
plan          | Plan → Execute → (Re-plan)       | 1 at step 0 only
ooda          | Observe → Orient → Decide → Act  | 1 per step

All stubs are deterministic and run without an API key.
When ANTHROPIC_API_KEY is set, real Claude calls are made with the same
fallback logic as agents/llm.py (a run is never crashed by an API error).

Usage (in rollout):
    from agents.multiagent import BLUE_MULTIAGENT_TYPES
    brain = BLUE_MULTIAGENT_TYPES["react"](n)
    aid = brain.step_decide(ctx)          # once per step
    brain.step_end(aid, reward, ctx)      # after env.step
    brain.episode_end()                   # after episode
"""
from __future__ import annotations

import json
import re
from typing import Optional

from agents import actions, llm
from agents.actions import BLUE_DECISION_N, BLUE_CATALOG

_MENU = actions.blue_menu_text()
_ACT_NAMES = [c[0] for c in BLUE_CATALOG[:BLUE_DECISION_N]]


# ══════════════════════════════════════════════════════════════ Base class ════

class BlueBrainBase:
    """Stateful blue commander — one instance per rollout episode."""

    def __init__(self, n: int):
        self.n = n                          # total drones in fleet
        self.t = 0                          # current step (advanced by step_end)
        self.memory: list[dict] = []        # per-step records

    # ── public interface ───────────────────────────────────────────────────

    def step_decide(self, ctx: dict) -> int:
        """Return ONE blue catalog id 0-8 for the whole team this step."""
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


# ══════════════════════════════════════════════════════════ 1. ReAct Blue ════

class ReActBlue(BlueBrainBase):
    """Observe → Reason (chain-of-thought) → Act each step.

    The LLM is given current state + recent history and asked to:
      1) Write one sentence of reasoning about the threat.
      2) Output  ACTION=<id>

    Without an API key the stub runs a threat-adaptive heuristic that mirrors
    the same observe→reason→act phases in pure Python.

    Observe  : compromised count, delta, threat level
    Reason   : (LLM) CoT; (stub) classify threat tier
    Act      : pick action for that tier
    """

    TIER_ACTION = {
        "safe":     2,   # Analyse
        "low":      1,   # Monitor
        "medium":   4,   # RetakeSuspicious
        "high":     3,   # RemoveSessions
        "critical": 4,   # RetakeSuspicious (aggressive)
    }

    def __init__(self, n: int, history_k: int = 5):
        super().__init__(n)
        self.history_k = history_k

    # ── Observe ─────────────────────────────────────────────────────────────
    def _observe(self, ctx: dict) -> dict:
        comp = ctx["compromised"]
        prev = self.memory[-1]["compromised"] if self.memory else 0
        frac = len(comp) / max(1, self.n)
        delta = len(comp) - prev
        spreading = delta > 0
        tier = ("safe" if frac == 0 else
                "low" if frac < 0.10 else
                "medium" if frac < 0.30 else
                "high" if frac < 0.55 else
                "critical")
        return {"count": len(comp), "frac": frac, "delta": delta,
                "spreading": spreading, "tier": tier}

    # ── Reason + Act (stub) ──────────────────────────────────────────────────
    def _stub(self, obs: dict) -> int:
        tier = obs["tier"]
        if obs["spreading"] and tier in ("medium", "high"):
            return 3    # Spreading → RemoveSessions first
        return self.TIER_ACTION[tier]

    # ── Reason + Act (LLM) ──────────────────────────────────────────────────
    def _build_prompt(self, obs: dict, ctx: dict) -> str:
        return (
            f"You are BLUE commander for a drone swarm.\n"
            f"[Observe] compromised={obs['count']}/{self.n}  "
            f"delta={obs['delta']:+d}  tier={obs['tier']}\n"
            f"{self._history_str(self.history_k)}\n\n"
            f"Step 1 – Reason: describe the threat in one sentence.\n"
            f"Step 2 – Act   : output exactly  ACTION=<id>  from:\n"
            f"{_MENU}"
        )

    def step_decide(self, ctx: dict) -> int:
        obs = self._observe(ctx)

        if not llm.available():
            return self._stub(obs)

        try:
            client = llm._get_client()
            msg = client.messages.create(
                model=llm.MODEL, max_tokens=80, temperature=0.0,
                system="Defense agent. Reason briefly, then output ACTION=<id>.",
                messages=[{"role": "user",
                           "content": self._build_prompt(obs, ctx)}])
            text = "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text")
            m = re.search(r"ACTION\s*=\s*(\d+)", text)
            if m:
                aid = int(m.group(1))
                if 0 <= aid < BLUE_DECISION_N:
                    return aid
            nums = [int(x) for x in re.findall(r"\b\d+\b", text)]
            for candidate in nums:
                if 0 <= candidate < BLUE_DECISION_N:
                    return candidate
        except Exception:
            pass
        return self._stub(obs)

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self.memory[-1]["tier"] = self._observe(ctx)["tier"]


# ══════════════════════════════════════════════════════════ 2. Reflect Blue ══

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


# ══════════════════════════════════════════════════════════ 3. Planner Blue ══

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


# ══════════════════════════════════════════════════════════ 4. OODA Blue ═════

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


# ══════════════════════════════════════════════════════════════ Registry ══════

BLUE_MULTIAGENT_TYPES: dict[str, type[BlueBrainBase]] = {
    "react":   ReActBlue,
    "reflect": ReflectBlue,
    "plan":    PlannerBlue,
    "ooda":    OODABlue,
}

ALL_BLUE_TYPES = ["rule", "llm", "rl"] + list(BLUE_MULTIAGENT_TYPES)
