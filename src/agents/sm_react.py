"""[구조 파악용] 모델: react   (O-P-E-R: 관찰-추론(CoT)-행동)
레지스트리 정의: ReActBlue(n, history_k=5)
※ 팀원 구조 파악 전용 — 실제 실행/임포트는 원본(src/agents/multiagent.py)을 사용.
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
