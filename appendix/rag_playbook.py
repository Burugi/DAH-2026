"""모델: rag   (플레이북 카드 최근접 검색(RAG-lite))
레지스트리 정의: RAGPlaybookBlue(n, guard_radius=30)
   BlueBrainBase 인터페이스: team_decide(ctx, agents)->per-drone action id 리스트,
   step_decide(ctx)->팀 단일 action, recovery_priority(ctx,agents,aids)->예산 우선순위.
"""
from __future__ import annotations
import math, json, re
from typing import Optional
# from agents.base import BlueBrainBase   # 상속 인터페이스(원본 참조)
class BlueBrainBase:                       # 최소 스텁
    def __init__(self, n): self.n = n


# 공용 헬퍼
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

class RAGPlaybookBlue(BlueBrainBase):
    """위협 인텔리전스 RAG — 시그니처로 사전 플레이북을 검색해 적용.

    '정찰 단계에서 군 정보 사전 검색' 아이디어의 오프라인 결정론 구현. 매 스텝 관측을
    5차원 시그니처로 압축하고, 사전 정의된 플레이북 카드(위협 프로파일 → stance/dispatch)
    중 가중 L1 최근접 카드를 검색해 적용한다. 검색된 카드 이름을 로그에 남겨 추론 추적 가능.

    시그니처 = [jam_frac, spoof_frac, comp_frac, growth, hub_loss]
    """

    # 카드: (이름, 시그니처 센트로이드, dispatch 함수 키)
    _CARDS = [
        ("quiet",         (0.0, 0.0, 0.0, 0.0, 0.0), "monitor"),
        ("jam_dominant",  (0.4, 0.0, 0.1, 0.0, 0.4), "anti_jam"),
        ("gps_stealth",   (0.0, 0.3, 0.1, 0.0, 0.1), "quarantine"),
        ("worm_cascade",  (0.1, 0.0, 0.3, 0.3, 0.2), "anti_worm"),
        ("mass_compromise", (0.1, 0.1, 0.7, 0.1, 0.5), "contain"),
    ]
    _W = (1.0, 1.0, 1.5, 1.5, 1.0)      # 시그니처 축별 가중치 (점령/확산 축 강조)

    def __init__(self, n: int, guard_radius: float = 30.0):
        super().__init__(n)
        self.guard_radius = guard_radius
        self._comp_hist: list[int] = []
        self._last_card: str = "quiet"

    def _signature(self, ctx: dict) -> tuple:
        comp = ctx["compromised"]
        snr = ctx.get("snr")
        gps = ctx.get("gps_err")
        link = ctx.get("link_up")
        n = self.n
        jam = sum(1 for i in range(n) if snr is not None and snr[i] <= 6) / n
        spoof = sum(1 for i in range(n) if gps is not None and gps[i] >= 8) / n
        cf = len(comp) / n
        self._comp_hist.append(len(comp))
        growth = 0.0
        if len(self._comp_hist) >= 2:
            growth = max(0, self._comp_hist[-1] - self._comp_hist[-2]) / n
        # hub_loss = 링크 끊긴 드론 비율
        hub_loss = (sum(1 for i in range(n) if link is not None and not link[i]) / n
                    if link is not None else 0.0)
        return (jam, spoof, cf, growth, hub_loss)

    def _retrieve(self, sig: tuple) -> str:
        best, best_d = self._CARDS[0], 1e18
        for card in self._CARDS:
            d = sum(self._W[k] * abs(sig[k] - card[1][k]) for k in range(5))
            if d < best_d:
                best, best_d = card, d
        self._last_card = best[0]
        return best[2]

    def _dispatch(self, policy: str, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        pos = ctx.get("pos")
        snr = ctx.get("snr")
        # 위협원 = 감염 + 강재밍(SNR<=6)
        threats = set(comp)
        if snr is not None:
            for i in range(self.n):
                if snr[i] <= 6:
                    threats.add(i)

        # (policy, role) → action id 매핑
        table = {
            "monitor":    {"comp": 3, "hub": 1, "guard": 4, "reserve": 1},
            "anti_jam":   {"comp": 3, "hub": 9, "guard": 7, "reserve": 7},
            "quarantine": {"comp": 3, "hub": 1, "guard": 6, "reserve": 1},
            "anti_worm":  {"comp": 3, "hub": 7, "guard": 4, "reserve": 4},
            "contain":    {"comp": 3, "hub": 9, "guard": 4, "reserve": 4},
        }[policy]

        aids = []
        for agent in agents:
            d = int(agent.split("_")[-1])
            if d in comp:
                aids.append(table["comp"])
                continue
            if not threats:
                aids.append(1)
                continue
            # guardian/reserve: 위협 근접도
            guard = True
            if pos is not None and d < len(pos):
                dmin = min((_dist(pos, d, j) for j in threats if j < len(pos)),
                           default=1e9)
                guard = dmin <= self.guard_radius
            aids.append(table["guard"] if guard else table["reserve"])
        return aids

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        sig = self._signature(ctx)
        policy = self._retrieve(sig)
        return self._dispatch(policy, ctx, agents)

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        self._signature(ctx)
        if not comp:
            return 1
        return 4 if len(comp) / self.n < 0.5 else 3

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self.memory[-1]["card"] = self._last_card
