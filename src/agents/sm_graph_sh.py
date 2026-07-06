"""[구조 파악용] 모델: graph_sh   (graph + 자가치유 우선 역이식(최종 권장))
레지스트리 정의: GraphCentralityBlue(n, n_hubs=3, selfheal=1.0)
※ 팀원 구조 파악 전용 — 실제 실행/임포트는 원본(src/agents/experimental.py)을 사용.
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


# 이 변형은 별도 클래스가 아니라 아래 base 클래스의 파라미터판입니다.
# 구조 = base 클래스 그대로, 등록만 다음과 같이:  GraphCentralityBlue(n, n_hubs=3, selfheal=1.0)

# ── 공용 헬퍼 ──
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

class GraphCentralityBlue(BlueBrainBase):
    """위치 근접 그래프 중심성 기반 방어 (GNN-lite).

    기존 hub 감지는 link_up(드론별 연결여부) 개수라 사실상 '연결됨' 이진값뿐이다.
    이 아키텍처는 pos + max_link로 근접 그래프 A[i,j]=1(dist<=max_link)을 만들고,
    degree + 2-hop 도달수를 중심성 대리값으로 쓴다. 연결성/가용성이 중요한 위협에서:

      · 감염 노드는 중심성이 높을수록(허브일수록) 우선 탈환/차단 — 연결 손실 최소화.
      · healthy 고중심성 허브는 Failsafe로 자율 연결 유지.
      · healthy 중간 노드는 감염 이웃을 탈환(guardian), 저중심성 leaf는 보존(Monitor).
    """

    def __init__(self, n: int, n_hubs: int = 3, selfheal: float = 0.0):
        super().__init__(n)
        self.n_hubs = min(n_hubs, n)
        # selfheal>0: whittle_sh와 동일한 자가치유 우선 항을 graph에 역이식(지식전이 일반성 검증).
        self.selfheal = selfheal
        self._prio = {}

    def _centrality(self, ctx: dict) -> list[float]:
        pos = ctx.get("pos")
        link = ctx.get("link_up")
        max_link = ctx.get("max_link", 40)
        if pos is None:
            return [0.0] * self.n
        m = min(self.n, len(pos))
        # 근접 인접행렬 (연결된 드론만 노드로)
        up = [bool(link[i]) if link is not None else True for i in range(m)]
        adj = [[False] * m for _ in range(m)]
        for i in range(m):
            if not up[i]:
                continue
            for j in range(i + 1, m):
                if up[j] and _dist(pos, i, j) <= max_link:
                    adj[i][j] = adj[j][i] = True
        deg = [sum(adj[i]) for i in range(m)]
        # 2-hop 도달수(중심성 대리값): degree + 이웃의 degree 합 가중
        cent = []
        for i in range(m):
            two_hop = set()
            for j in range(m):
                if adj[i][j]:
                    two_hop.add(j)
                    for k in range(m):
                        if adj[j][k]:
                            two_hop.add(k)
            two_hop.discard(i)
            cent.append(deg[i] + 0.5 * len(two_hop))
        cent += [0.0] * (self.n - m)
        return cent

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        cent = self._centrality(ctx)
        # healthy 고중심성 노드 = 허브
        healthy = [d for d in range(self.n) if d not in comp]
        hubs = set(sorted(healthy, key=lambda d: -cent[d])[:self.n_hubs])
        nbr = _proximity_graph(dict(ctx, n=self.n))
        spread_pot = {c: len(nbr.get(c, set()) - comp) for c in comp}

        self._prio = {}
        aids = []
        for agent in agents:
            d = int(agent.split("_")[-1])
            if d in comp:
                aids.append(3)                    # 감염 → 세션 제거 (탈환은 healthy가 수행)
                if self.selfheal:
                    # 자가치유 우선순위 = 자신의 전파잠재력 × selfheal (whittle_sh와 동일 원리)
                    self._prio[d] = self.selfheal * spread_pot.get(d, 0)
                continue
            if not comp:
                aids.append(1)
                continue
            if d in hubs:
                aids.append(9)                    # 허브 → Failsafe(자율 연결 유지)
            elif cent[d] > 0:
                aids.append(4)                    # 중간 노드 → 고중심성 감염 탈환
                if self.selfheal:                 # 원판 graph는 중립(예산 정렬 원순서) 유지
                    self._prio[d] = cent[d] * 0.01
            else:
                aids.append(1)                    # 고립 leaf → 보존
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        if not comp:
            return 1
        return 4 if len(comp) / self.n < 0.5 else 3
