"""Hierarchical Hybrid Swarm Defense — Blue Brain.

Architecture (Gemini proposal, adapted for CybORG CC3):

  ┌─────────────────────────────────────────┐
  │   Swarm Commander  (Claude LLM)         │  event-driven, ~4-5 calls / 40 steps
  │   → decides Stance + drone weights      │
  └────────────────┬────────────────────────┘
                   │ stance, weights
  ┌────────────────▼────────────────────────┐
  │   Stance Controller  (rule-based fast)  │  every step, no LLM
  │   → maps (stance, role) → action_id     │
  └────────────────┬────────────────────────┘
                   │ per-drone action ids
  ┌───────────┬────▼──────┬─────────────────┐
  │  Hub UAVs │ Leaf UAVs │ UGVs / Compr.   │
  └───────────┴───────────┴─────────────────┘

LLM call triggers (Event-Driven):
  - step 0  (initial stance)
  - 3+ new compromises since last call
  - any hub drone loses connectivity
  - stance objective achieved (auto-downgrade to NORMAL)

Stances:
  NORMAL     standard monitoring + retake
  ANTI_JAM   restore connectivity (AllowTraffic everywhere)
  QUARANTINE isolate compromised / GPS-spoofed drones
  ANTI_WORM  aggressive retake + session removal
  EMERGENCY  all drones RemoveSessions simultaneously
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

from agents import llm
from agents.base import BlueBrainBase

# LLM model for commander — can override via LAB_LLM_MODEL env var.
# Default: same model as llm.py (haiku without key, sonnet when key set).
_COMMANDER_MODEL = "claude-sonnet-4-6"

STANCES = ("NORMAL", "ANTI_JAM", "QUARANTINE", "ANTI_WORM", "EMERGENCY", "CONTAIN")

# (stance, role) → blue catalog action id
# role ∈ {hub, compromised, leaf, safe}
_POLICY: dict[tuple[str, str], int] = {
    ("NORMAL",     "hub"):         1,   # Monitor — stay aware
    ("NORMAL",     "compromised"): 3,   # RemoveSessions — clean own node
    ("NORMAL",     "leaf"):        4,   # RetakeSuspicious
    ("NORMAL",     "safe"):        1,   # Monitor
    ("ANTI_JAM",   "hub"):         9,   # Failsafe — autonomous local defense when hub is jammed/cut
    ("ANTI_JAM",   "compromised"): 3,   # RemoveSessions (still needed)
    ("ANTI_JAM",   "leaf"):        7,   # AllowTraffic — contribute to mesh
    ("ANTI_JAM",   "safe"):        7,   # AllowTraffic
    ("QUARANTINE", "hub"):         1,   # Monitor — hubs stay connected
    ("QUARANTINE", "compromised"): 3,   # RemoveSessions
    ("QUARANTINE", "leaf"):        6,   # BlockSuspicious — quarantine edges
    ("QUARANTINE", "safe"):        1,   # Monitor
    ("ANTI_WORM",  "hub"):         7,   # AllowTraffic — keep mesh alive
    ("ANTI_WORM",  "compromised"): 3,   # RemoveSessions — urgent
    ("ANTI_WORM",  "leaf"):        4,   # RetakeSuspicious — recapture
    ("ANTI_WORM",  "safe"):        4,   # RetakeSuspicious — pre-empt spread
    ("EMERGENCY",  "hub"):         3,   # RemoveSessions — all hands
    ("EMERGENCY",  "compromised"): 3,   # RemoveSessions
    ("EMERGENCY",  "leaf"):        3,   # RemoveSessions
    ("EMERGENCY",  "safe"):        3,   # RemoveSessions
    # CONTAIN — 대량 점령 시 연결성 보존형 대응 (EMERGENCY 대체, ablation 근거)
    # EMERGENCY(전원 RemoveSessions)는 가용성을 버려 유의하게 해로움(p=3.2e-4).
    # 허브는 Failsafe로 자율 연결 유지, 나머지는 공격적 탈환/제거.
    ("CONTAIN",    "hub"):         9,   # Failsafe — 허브 자율 방어(연결 유지)
    ("CONTAIN",    "compromised"): 3,   # RemoveSessions
    ("CONTAIN",    "leaf"):        4,   # RetakeSuspicious
    ("CONTAIN",    "safe"):        4,   # RetakeSuspicious
}


class HierarchicalBlue(BlueBrainBase):
    """Hierarchical Hybrid Swarm Defense blue agent.

    One instance per rollout episode. Exposes:
      step_decide(ctx)            → single representative action id (for logging)
      team_decide(ctx, agents)    → per-drone action id list (for run.py)
    """

    def __init__(self, n: int, n_hubs: int = 4, llm_budget: int = 5):
        super().__init__(n)
        self.n_hubs = min(n_hubs, n)
        self.llm_budget = llm_budget
        self.llm_calls = 0
        self.stance: str = "NORMAL"
        self.drone_weights: list[float] = [1.0] * n
        self._hub_ids: list[int] = list(range(self.n_hubs))
        self._last_call_comp: int = 0       # compromised count at last LLM call
        self._last_call_t: int = -99

    # ── Hub detection ──────────────────────────────────────────────────────────

    def _detect_hubs(self, ctx: dict) -> list[int]:
        """Return top-K drones by live link count (connectivity degree)."""
        link = ctx.get("link_up")           # shape (n,) at current step
        if link is None:
            return list(range(self.n_hubs))
        # Degree = number of links still up per drone
        degrees = {i: int(link[i]) for i in range(self.n)}
        return sorted(degrees, key=lambda x: -degrees[x])[:self.n_hubs]

    # ── Role classification ────────────────────────────────────────────────────

    def _get_role(self, drone_id: int, ctx: dict) -> str:
        if drone_id in ctx["compromised"]:
            return "compromised"
        if drone_id in self._hub_ids:
            return "hub"
        if ctx["compromised"]:
            return "leaf"
        return "safe"

    # ── State abstraction ──────────────────────────────────────────────────────

    def _abstract_state(self, ctx: dict) -> dict:
        """Compress raw obs to a tiny JSON the LLM can parse quickly."""
        comp = ctx["compromised"]
        link = ctx.get("link_up")
        snr  = ctx.get("snr")
        gps  = ctx.get("gps_err")

        hub_status = {}
        for h in self._hub_ids:
            hub_status[f"drone_{h}"] = {
                "connected":   bool(link[h])    if link is not None else True,
                "snr_ok":      bool(snr[h] > 6) if snr  is not None else True,
                "gps_ok":      bool(gps[h] < 8) if gps  is not None else True,
                "compromised": h in comp,
            }

        jammed  = [i for i in range(self.n) if snr is not None and snr[i] <= 6]
        spoofed = [i for i in range(self.n) if gps is not None and gps[i] >= 8]

        return {
            "step":                 self.t,
            "total_drones":        self.n,
            "compromised_count":   len(comp),
            "compromised_ids":     sorted(comp),
            "hub_status":          hub_status,
            "jammed_drones":       jammed,
            "gps_spoofed_drones":  spoofed,
            "network_availability": round(1 - len(comp) / max(1, self.n), 2),
            "current_stance":      self.stance,
        }

    # ── LLM trigger ───────────────────────────────────────────────────────────

    def _should_call_llm(self, ctx: dict) -> bool:
        if self.llm_calls >= self.llm_budget:
            return False
        if self.t == 0:
            return True
        comp = len(ctx["compromised"])
        if comp - self._last_call_comp >= 3:
            return True
        link = ctx.get("link_up")
        if link is not None:
            for h in self._hub_ids:
                if not link[h]:
                    return True
        return False

    # ── Claude Commander ───────────────────────────────────────────────────────

    def _call_commander(self, abstract: dict) -> dict:
        if not llm.available():
            return self._stub_commander(abstract)
        try:
            import anthropic
            client = anthropic.Anthropic()
            system = (
                "You are the commander of a drone swarm cyber defense system.\n"
                "Analyze the JSON network state and choose a defense stance.\n"
                "Reply with ONLY valid JSON — no explanation, no markdown:\n"
                '{"analysis":"<one sentence>","stance":"<STANCE>","drone_weights":[<floats>]}\n'
                f"STANCE ∈ {list(STANCES)}.  drone_weights length = {self.n}.\n"
                "weight meaning: 2.0=high priority recover/protect, 1.0=normal, 0.3=deprioritize.\n"
                "Stances: NORMAL=monitor+retake, ANTI_JAM=restore links, "
                "QUARANTINE=block edges, ANTI_WORM=aggressive retake, EMERGENCY=full RemoveSessions."
            )
            msg = client.messages.create(
                model=_COMMANDER_MODEL, max_tokens=300, temperature=0.0,
                system=system,
                messages=[{"role": "user",
                           "content": json.dumps(abstract, indent=2)}])
            text = "".join(b.text for b in msg.content
                           if getattr(b, "type", "") == "text")
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                stance  = result.get("stance", "NORMAL")
                if stance not in STANCES:
                    stance = "NORMAL"
                weights = result.get("drone_weights", [1.0] * self.n)
                if len(weights) != self.n:
                    weights = [1.0] * self.n
                return {"stance": stance, "weights": [float(w) for w in weights]}
        except Exception:
            pass
        return self._stub_commander(abstract)

    def _stub_commander(self, state: dict) -> dict:
        """Offline stance selection: mirrors what Claude would do without API."""
        comp  = state["compromised_count"]
        n     = state["total_drones"]
        jammed  = len(state.get("jammed_drones", []))
        spoofed = len(state.get("gps_spoofed_drones", []))

        if comp / max(1, n) > 0.5:
            stance = "EMERGENCY"
        elif jammed >= self.n_hubs:          # all hubs jammed → connectivity crisis
            stance = "ANTI_JAM"
        elif comp >= 3:
            stance = "ANTI_WORM"
        elif jammed >= 2 or spoofed >= 2:
            stance = "QUARANTINE"
        else:
            stance = "NORMAL"

        weights = [1.0] * n
        for i in state.get("jammed_drones", []):
            if i < n:
                weights[i] = 0.3            # deprioritize jammed (can't help)
        for i in state.get("compromised_ids", []):
            if i < n:
                weights[i] = 2.0            # high priority to recover

        return {"stance": stance, "weights": weights}

    # ── Main decision interface ────────────────────────────────────────────────

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        """Per-drone action ids for all live agents this step.

        Called by run.py instead of step_decide() when blue_brain supports it.
        """
        self._hub_ids = self._detect_hubs(ctx)

        if self._should_call_llm(ctx):
            abstract = self._abstract_state(ctx)
            result   = self._call_commander(abstract)
            self.stance        = result["stance"]
            self.drone_weights = result["weights"]
            self._last_call_comp = len(ctx["compromised"])
            self._last_call_t    = self.t
            self.llm_calls      += 1

        aids = []
        for agent in agents:
            drone_id = int(agent.split("_")[-1])
            role = self._get_role(drone_id, ctx)
            aids.append(_POLICY.get((self.stance, role), 1))

        return aids

    def step_decide(self, ctx: dict) -> int:
        """Fallback / representative action id (used for logging and non-hierarchical runs)."""
        self._hub_ids = self._detect_hubs(ctx)
        if self._should_call_llm(ctx):
            result = self._call_commander(self._abstract_state(ctx))
            self.stance        = result["stance"]
            self.drone_weights = result["weights"]
            self._last_call_comp = len(ctx["compromised"])
            self.llm_calls      += 1
        comp = ctx["compromised"]
        if not comp:
            return 1    # Monitor
        if len(comp) / self.n > 0.5:
            return 3    # RemoveSessions (emergency)
        return 4        # RetakeSuspicious

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self.memory[-1]["stance"]    = self.stance
        self.memory[-1]["llm_calls"] = self.llm_calls


# ══════════════════════════════════════════════ Ablation variants ══════════════

class HierNoStance(HierarchicalBlue):
    """Ablation A: per-drone role dispatch, but stance is always NORMAL.

    Isolates the contribution of per-drone dispatch alone (no stance switching).
    """
    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        self._hub_ids = self._detect_hubs(ctx)
        self.stance = "NORMAL"          # frozen — never changes
        self.llm_calls = 0
        aids = []
        for agent in agents:
            drone_id = int(agent.split("_")[-1])
            role = self._get_role(drone_id, ctx)
            aids.append(_POLICY.get(("NORMAL", role), 1))
        return aids


class HierNoTrigger(HierarchicalBlue):
    """Ablation B: per-drone dispatch + stance, updated EVERY step (no event filter).

    Isolates the contribution of event-driven triggering vs. always-on stance update.
    llm_budget set to episode length so it never runs out.
    """
    def __init__(self, n: int, n_hubs: int = 4):
        super().__init__(n, n_hubs=n_hubs, llm_budget=999)

    def _should_call_llm(self, ctx: dict) -> bool:
        return True     # update stance every step


class HierDropStance(HierarchicalBlue):
    """Leave-one-stance-out: remap one non-NORMAL stance to NORMAL.

    Full hier_h2 minus this variant's D_mult = the dropped stance's marginal
    contribution. Isolates which individual stance drives (or hurts) performance.
    """
    def __init__(self, n: int, drop: str, n_hubs: int = 2, llm_budget: int = 5):
        super().__init__(n, n_hubs=n_hubs, llm_budget=llm_budget)
        self.drop = drop

    def _stub_commander(self, state: dict) -> dict:
        result = super()._stub_commander(state)
        if result["stance"] == self.drop:
            result["stance"] = "NORMAL"
        return result


class HierH3(HierarchicalBlue):
    """동적 역할 배분 (위협 근접도 기반) — dispatch 정밀화.

    ablation 결과 성능 동력은 stance가 아니라 per-drone dispatch였다(+1.6%). hier_h2는
    모든 healthy non-hub 드론을 일률적으로 'leaf'(RetakeSuspicious)로 처리한다. h3는
    이를 위협 근접도로 분화한다:

      · guardian  : 위협(점령/재밍 드론) 반경 내 healthy 드론 → 능동 탈환/차단
      · reserve   : 위협에서 먼 healthy 드론 → Monitor로 보존(행동낭비·공격면 축소)

    허브/점령 드론의 행동은 hier_h2와 동일(stance 기반)하게 유지해, 순수하게 dispatch
    정밀화의 효과만 분리 측정한다.
    """

    def __init__(self, n: int, n_hubs: int = 2, llm_budget: int = 5,
                 guard_radius: Optional[float] = None):
        super().__init__(n, n_hubs=n_hubs, llm_budget=llm_budget)
        self.guard_radius = guard_radius   # None → ctx의 max_link 사용

    def _threat_ids(self, ctx: dict) -> set:
        """점령됐거나 강하게 재밍된(SNR<=6) 드론 = 위협원."""
        threats = set(ctx["compromised"])
        snr = ctx.get("snr")
        if snr is not None:
            for i in range(self.n):
                if snr[i] <= 6:
                    threats.add(i)
        return threats

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        self._hub_ids = self._detect_hubs(ctx)
        if self._should_call_llm(ctx):
            result = self._call_commander(self._abstract_state(ctx))
            self.stance          = result["stance"]
            self.drone_weights   = result["weights"]
            self._last_call_comp = len(ctx["compromised"])
            self._last_call_t    = self.t
            self.llm_calls      += 1

        comp    = ctx["compromised"]
        pos     = ctx.get("pos")
        radius  = self.guard_radius or ctx.get("max_link", 40)
        threats = self._threat_ids(ctx)

        aids = []
        for agent in agents:
            d = int(agent.split("_")[-1])
            if d in comp:
                aids.append(_POLICY.get((self.stance, "compromised"), 3))
                continue
            if d in self._hub_ids:
                aids.append(_POLICY.get((self.stance, "hub"), 1))
                continue
            if not threats:
                aids.append(1)                       # 위협 없음 → 감시
                continue
            # healthy non-hub: 위협 근접도로 guardian/reserve 분화
            guardian = True
            if pos is not None and d < len(pos):
                dmin = min((math.hypot(pos[d][0] - pos[j][0], pos[d][1] - pos[j][1])
                            for j in threats if j < len(pos)), default=1e9)
                guardian = dmin <= radius
            if guardian:
                aids.append(_POLICY.get((self.stance, "leaf"), 4))   # 능동 대응
            else:
                aids.append(1)                       # reserve → Monitor 보존
        return aids


class HierV2(HierarchicalBlue):
    """증거 기반 개선판 (stance ablation 결과 반영).

    변경점:
      1) EMERGENCY → CONTAIN : 대량 점령 시 전원 RemoveSessions 대신 허브 연결을
         Failsafe로 유지. EMERGENCY가 유의하게 해로웠음(Wilcoxon p=3.2e-4).
      2) ANTI_WORM 게이팅 : ANTI_WORM은 연결성/가용성엔 크게 도움(+0.026/+0.008)이나
         순수 점령·탐지회피 시나리오엔 해로움. 재밍이 동반될 때만(=연결성 위협 신호)
         ANTI_WORM을 쓰고, 그 외 대량 점령은 NORMAL(공격적 탈환)로 처리.
    """
    def _stub_commander(self, state: dict) -> dict:
        comp = state["compromised_count"]
        n = state["total_drones"]
        jammed = len(state.get("jammed_drones", []))
        spoofed = len(state.get("gps_spoofed_drones", []))

        if comp / max(1, n) > 0.5:
            stance = "CONTAIN"                       # (1) EMERGENCY 대체
        elif jammed >= self.n_hubs:
            stance = "ANTI_JAM"
        elif comp >= 3:
            # (2) ANTI_WORM은 연결성 스트레스(재밍 동반)일 때만
            stance = "ANTI_WORM" if jammed >= 1 else "NORMAL"
        elif jammed >= 2 or spoofed >= 2:
            stance = "QUARANTINE"
        else:
            stance = "NORMAL"

        weights = [1.0] * n
        for i in state.get("jammed_drones", []):
            if i < n:
                weights[i] = 0.3
        for i in state.get("compromised_ids", []):
            if i < n:
                weights[i] = 2.0
        return {"stance": stance, "weights": weights}
