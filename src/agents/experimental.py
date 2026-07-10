"""실험용 신규 Blue 방어 아키텍처 (오프라인 결정론적, API 키 불필요).

세 아키텍처는 각기 다른 위협 클래스를 노려 상보성(complementarity)을 만든다.
모두 BlueBrainBase를 상속하고, 성능 동력인 per-drone dispatch를 위해 team_decide로
드론별 action id를 반환한다. 점수는 정본 score.d_mult_single(comp_F1 제외)로 측정.

  Predictive       SIR형 확산 예측 → 감염 후보 선제 차단/탈환   (점령·확산)
  GraphCentrality  위치 근접 그래프 중심성 → 허브 우선 보호       (연결성·가용성)
  RAGPlaybook      위협 시그니처 → 사전 플레이북 KB 검색 후 적용  (탐지회피·가용성)

action id (BLUE_CATALOG 0-9):
  1 Monitor  2 Analyse  3 RemoveSessions  4 RetakeSuspicious  5 RetakeRandom
  6 BlockSuspicious  7 AllowTraffic  9 Failsafe
"""
from __future__ import annotations

import math
from typing import Optional

from agents.base import BlueBrainBase


def _dist(pos, i, j):
    return math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])


# ══════════════════════════════════════════════ 1. Predictive ══════════════════

class PredictiveBlue(BlueBrainBase):
    """SIR형 확산 예측 기반 선제 방어.

    웜은 감염 드론에서 근접한 healthy 드론으로 전파된다(run._worm_step: proximity).
    따라서 '다음에 감염될 드론' = 감염원 반경 내 healthy 드론이다. 이 아키텍처는:

      · 감염원(spreader) 중 healthy 이웃이 많은 노드를 우선 차단(BlockSuspicious)해
        전파 front를 끊는다.
      · front에 인접한 healthy 드론은 guardian으로 감염 이웃을 능동 탈환(RetakeSuspicious).
      · front에서 먼 healthy 드론은 reserve로 보존(Monitor) — 공격면·행동낭비 축소.

    관측된 확산 속도(최근 Δcompromised)로 guardian 반경을 동적 조절한다:
    확산이 가속 중이면 반경을 넓혀 공격적으로, 안정적이면 좁혀 보존적으로.
    """

    def __init__(self, n: int, base_radius: float = 25.0, block_top: int = 2):
        super().__init__(n)
        self.base_radius = base_radius
        self.block_top = block_top          # 매 스텝 차단할 상위 spreader 수
        self._comp_hist: list[int] = []

    def _spread_rate(self) -> int:
        """최근 2스텝 감염 증가량(양수면 확산 중)."""
        h = self._comp_hist
        if len(h) < 2:
            return 0
        return h[-1] - h[-2]

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        pos = ctx.get("pos")
        max_link = ctx.get("max_link", 40)
        self._comp_hist.append(len(comp))

        # 확산 가속 시 반경 확대(1.4배), 안정 시 축소(0.7배)
        rate = self._spread_rate()
        radius = self.base_radius * (1.4 if rate > 0 else 0.7 if len(comp) else 1.0)

        # 감염원별 전파 잠재력 = 반경 내 healthy 이웃 수 → 상위 spreader 선정
        spreaders: set = set()
        if pos is not None and comp:
            potential = {}
            for c in comp:
                if c >= len(pos):
                    continue
                nb = sum(1 for d in range(self.n)
                         if d not in comp and d < len(pos) and _dist(pos, c, d) <= radius)
                potential[c] = nb
            spreaders = set(sorted(potential, key=lambda x: -potential[x])[:self.block_top])

        aids = []
        for agent in agents:
            d = int(agent.split("_")[-1])
            if d in comp:
                # 감염 노드: 전파력 큰 spreader면 차단, 아니면 세션 제거
                aids.append(6 if d in spreaders else 3)
                continue
            if not comp or pos is None or d >= len(pos):
                aids.append(1)                    # 위협 없음 → 감시
                continue
            dmin = min((_dist(pos, d, c) for c in comp if c < len(pos)), default=1e9)
            if dmin <= radius:
                aids.append(4)                    # front 인접 guardian → 능동 탈환
            else:
                aids.append(1)                    # reserve → 보존
        return aids

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        self._comp_hist.append(len(comp))
        if not comp:
            return 1
        return 6 if len(comp) / self.n < 0.5 else 3


# ══════════════════════════════════════════ 2. GraphCentrality ═════════════════

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


# ══════════════════════════════════════════════ 3. RAGPlaybook ═════════════════

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


# ══════════════════════════════════════ 공용 헬퍼 (아래 4종 공유) ═══════════════

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


# ══════════════════════════════════════════════ 4. BanditFeedback ══════════════

class BanditFeedbackBlue(BlueBrainBase):
    """온라인 컨텍스추얼 밴딧 (RLHF-lite) — step reward 피드백으로 정책 학습.

    상태 버킷(감염 비율 tier) × arm(팀 dispatch 정책)의 가치 Q(bucket, arm)를 에피소드
    내에서 증분 평균으로 학습하고, ε-greedy로 선택한다. 매 스텝 받은 보상을 피드백으로
    직전에 고른 arm의 Q를 갱신 → 관측→행동→보상→갱신의 피드백 루프.

    arm: aggressive / centrality / conservative / remove_all / monitor
    """

    ARMS = ["aggressive", "centrality", "conservative", "remove_all", "monitor"]

    def __init__(self, n: int, eps: float = 0.15):
        super().__init__(n)
        self.eps = eps
        self.q: dict = {}
        self.cnt: dict = {}
        self._last: Optional[tuple] = None
        self._graph = GraphCentralityBlue(n)      # centrality arm 위임

    def _bucket(self, ctx: dict) -> int:
        f = len(ctx["compromised"]) / max(1, self.n)
        return 0 if f == 0 else 1 if f < 0.15 else 2 if f < 0.4 else 3

    def _select(self, bucket: int) -> str:
        import random
        # 미시도 arm 우선 탐색
        unseen = [a for a in self.ARMS if (bucket, a) not in self.cnt]
        if unseen:
            return random.choice(unseen)
        if random.random() < self.eps:
            return random.choice(self.ARMS)
        return max(self.ARMS, key=lambda a: self.q.get((bucket, a), 0.0))

    def _dispatch(self, arm: str, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        if arm == "centrality":
            return self._graph.team_decide(ctx, agents)
        threats = _threat_set(ctx)
        radius = ctx.get("max_link", 40)
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            if arm == "remove_all":
                aids.append(3)
            elif arm == "monitor":
                aids.append(1)
            elif arm == "aggressive":
                aids.append(4 if comp else 1)
            else:  # conservative
                aids.append(4 if _near_threat(d, ctx, threats, radius) else 1)
        return aids

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        b = self._bucket(ctx)
        arm = self._select(b)
        self._last = (b, arm)
        return self._dispatch(arm, ctx, agents)

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        if self._last is not None:
            k = self._last
            self.cnt[k] = self.cnt.get(k, 0) + 1
            self.q[k] = self.q.get(k, 0.0) + (reward - self.q.get(k, 0.0)) / self.cnt[k]
            self.memory[-1]["arm"] = k[1]

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)


# ══════════════════════════════════════════════ 5. EnsembleMoE ═════════════════

class EnsembleMoEBlue(BlueBrainBase):
    """전문가 혼합(MoE) — graph·predictive·rule 전문가의 드론별 제안을 봉쇄우선 결합.

    세 전문가가 각자 per-drone action을 제안하면, 드론마다 '봉쇄 우선순위'가 가장 높은
    행동을 채택(게이팅). 서로 다른 전문성을 한 팀 행동으로 합성한다.
    """

    # 동점 시 선호 순위(앞일수록 우선) — graph 전문가 성향 반영
    _PRIORITY = [4, 3, 9, 6, 5, 7, 2, 1, 0]

    def __init__(self, n: int):
        super().__init__(n)
        self._graph = GraphCentralityBlue(n)
        self._pred = PredictiveBlue(n)

    def _rule_aids(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        return [3 if int(a.split("_")[-1]) in comp else (4 if comp else 1)
                for a in agents]

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        votes = [self._graph.team_decide(ctx, agents),
                 self._pred.team_decide(ctx, agents),
                 self._rule_aids(ctx, agents)]
        rank = lambda x: self._PRIORITY.index(x) if x in self._PRIORITY else 99
        out = []
        for i in range(len(agents)):
            cand = [v[i] for v in votes]
            # 다수결; 동점이면 선호 순위로 결정
            counts = {}
            for c in cand:
                counts[c] = counts.get(c, 0) + 1
            top = max(counts.values())
            tied = [c for c in counts if counts[c] == top]
            out.append(min(tied, key=rank))
        return out

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)


# ══════════════════════════════════════════════ 6. HybridCore ══════════════════

class HybridCoreBlue(BlueBrainBase):
    """rule 우선순위 + rl 가치추정 + llm 위협추론을 단일 코어로 융합 (ideation C3).

      · rule  : 자기 감염 → RemoveSessions (고정 우선순위)
      · llm   : 감염 비율 tier로 공격성 추론
      · rl    : 최근 보상 EMA(value)가 나쁘면 더 공격적으로 전환
    healthy 드론은 (위협 근접 or 고tier or 저value)면 능동 탈환, 아니면 분석(Analyse).
    """

    def __init__(self, n: int):
        super().__init__(n)
        self._val = 0.0        # reward EMA (rl-ish value)

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        f = len(comp) / max(1, self.n)
        tier = 0 if f == 0 else 1 if f < 0.15 else 2 if f < 0.4 else 3
        threats = _threat_set(ctx)
        radius = ctx.get("max_link", 40)
        aggressive = tier >= 2 or self._val < -8.0
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            if not comp:
                aids.append(1)
                continue
            if aggressive or _near_threat(d, ctx, threats, radius):
                aids.append(4)
            else:
                aids.append(2)      # Analyse (약한 대응)
        return aids

    def step_end(self, aid: int, reward: float, ctx: dict) -> None:
        super().step_end(aid, reward, ctx)
        self._val = 0.7 * self._val + 0.3 * reward

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)


# ══════════════════════════════════════════════ 7. Debate ══════════════════════

class DebateBlue(BlueBrainBase):
    """다중 페르소나 토론/자기일관성 — Hawk(최대 봉쇄) vs Dove(보존) 제안 → 심판 채택.

    두 페르소나가 드론별 행동을 제안하고, 확산 여부·위협 tier로 심판이 드론별로 채택한다.
    확산 중이거나 고tier면 Hawk(능동 탈환), 아니면 Dove(위협 근접 시만 탈환, 그 외 보존).
    """

    def __init__(self, n: int):
        super().__init__(n)

    def team_decide(self, ctx: dict, agents: list[str]) -> list[int]:
        comp = ctx["compromised"]
        prev = self.memory[-1]["compromised"] if self.memory else 0
        spreading = len(comp) > prev
        f = len(comp) / max(1, self.n)
        threats = _threat_set(ctx)
        radius = ctx.get("max_link", 40)
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            if not comp:
                aids.append(1)
                continue
            hawk = 4
            dove = 4 if _near_threat(d, ctx, threats, radius) else 1
            aids.append(hawk if (spreading or f >= 0.4) else dove)
        return aids

    def step_decide(self, ctx: dict) -> int:
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp) / self.n < 0.5 else 3)


# ══════════════════════ 할당형 (예산 제약 하 "누구를 탈환하나"를 잘 푸는 부류) ═════
# 공통: recovery_priority(ctx, agents, aids) → live 에이전트별 우선순위 점수 리스트.
#       run.rollout이 recovery_budget k 초과 시 이 점수 상위 k개만 회복행동을 유지.

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


class AuctionBlue(BlueBrainBase):
    """시장기반 태스크 경매 (CBBA류).

    감염 드론 = 태스크, 건강 드론 = 입찰자. 입찰가 = f(거리, SNR). 그리디 순차경매로
    각 태스크를 최고 입찰 건강 드론에 1:1 배정. 낙찰자는 Retake(4), 유찰 태스크 인접
    건강 드론은 Block(6), 나머지 Monitor(1). 예산 하에서 낙찰 입찰가가 곧 우선순위.
    """
    def __init__(self, n: int):
        super().__init__(n)
        self._bid = {}      # drone -> 낙찰 입찰가 (recovery_priority용)

    def _auction(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos"); snr = ctx.get("snr")
        healthy = [int(a.split("_")[-1]) for a in agents
                   if int(a.split("_")[-1]) not in comp]
        tasks = list(comp)
        assign = {}                     # drone -> True(낙찰)
        self._bid = {}
        if pos is not None:
            avail = set(healthy)
            for task in tasks:
                if task >= len(pos):
                    continue
                best, bestbid = None, -1.0
                for d in avail:
                    if d >= len(pos):
                        continue
                    dd = _dist(pos, d, task)
                    bid = 1.0 / (1.0 + dd) * (1.0 + max(0.0, (snr[d] if snr is not None else 20) / 20.0))
                    if bid > bestbid:
                        best, bestbid = d, bid
                if best is not None:
                    assign[best] = True; self._bid[best] = bestbid; avail.discard(best)
        return assign

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos")
        assign = self._auction(ctx, agents)
        nbr = _proximity_graph(dict(ctx, n=self.n))
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
            elif assign.get(d):
                aids.append(4)                          # 낙찰 → 탈환
            elif nbr.get(d, set()) & comp:
                aids.append(6)                          # 유찰 태스크 인접 → 차단
            else:
                aids.append(1)
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._bid.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)


class WhittleBlue(BlueBrainBase):
    """휘틀/cμ 인덱스 스케줄러 — 학습 없는 폐형 우선순위.

    감염 드론(대기 작업)의 인덱스 = 다음 스텝 기대 전염수(반경 내 건강 이웃 수) /
    기대 탈환비용(최근접 건강 드론 거리). 건강 드론(서버)을 인덱스 큰 작업부터 배정.
    """
    def __init__(self, n: int, selfheal: float = 0.0):
        super().__init__(n)
        self._prio = {}
        # selfheal>0: 감염 드론 자가치유(RemoveSessions)에 전파잠재력 기반 우선순위 부여.
        # 기본 0(원판)은 자가치유 우선순위=0이라 예산 하에서 굶김 → k=3 딥의 원인(가설).
        # EvoBlue가 k=6에서 자가치유 우선을 스스로 발견한 것을 whittle에 역이식(지식전이).
        self.selfheal = selfheal

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos")
        nbr = _proximity_graph(dict(ctx, n=self.n))
        healthy = [int(a.split("_")[-1]) for a in agents
                   if int(a.split("_")[-1]) not in comp]
        spread_pot = {c: len(nbr.get(c, set()) - comp) for c in comp}   # 감염원 전파잠재력
        # 작업 인덱스
        index = {}
        for c in comp:
            if pos is not None and c < len(pos) and healthy:
                cost = min((_dist(pos, c, d) for d in healthy if d < len(pos)), default=1e9)
            else:
                cost = 1.0
            index[c] = spread_pot[c] / (1.0 + cost)
        # 인덱스 큰 작업부터 최근접 건강 드론 배정
        assign = {}; self._prio = {}
        avail = set(healthy)
        for c in sorted(index, key=lambda x: -index[x]):
            if pos is None or c >= len(pos) or not avail:
                continue
            d = min(avail, key=lambda x: _dist(pos, x, c) if x < len(pos) else 1e9)
            assign[d] = True; self._prio[d] = index[c]; avail.discard(d)
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                if self.selfheal:
                    # 자가치유 우선순위 = 자신의 전파잠재력 × selfheal (탈환 인덱스와 동일 스케일)
                    self._prio[d] = self.selfheal * spread_pot.get(d, 0)
            elif assign.get(d):
                aids.append(4)
            else:
                aids.append(1)
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)


class MinCutBlue(BlueBrainBase):
    """그래프 최소절단 방화벽 (근사).

    감염집합과 건강집합 사이의 경계(프론티어)를 위상적으로 끊는다. 프론티어 = 감염
    드론과 인접한 건강 드론. 프론티어 건강 드론은 감염 이웃 수가 많을수록 우선 탈환/차단
    (절단 효과 큼). 그 외 건강 드론은 벽 뒤에서 보존(Monitor).
    """
    def __init__(self, n: int):
        super().__init__(n)
        self._prio = {}

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]
        nbr = _proximity_graph(dict(ctx, n=self.n))
        self._prio = {}
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            inf_nb = nbr.get(d, set()) & comp
            if inf_nb:                                   # 프론티어 건강 드론
                self._prio[d] = float(len(inf_nb))       # 절단 효과 = 감염 이웃 수
                # 감염 이웃이 많으면 탈환(4), 하나면 차단(6)으로 간선 제거
                aids.append(4 if len(inf_nb) >= 2 else 6)
            else:
                aids.append(1)                           # 벽 뒤 → 보존
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)


# ══════════════════════ Phase 2: 메커니즘 다양성 (env 무변경) ═══════════════════

class RiskFieldBlue(BlueBrainBase):
    """베이지안 위험장 선제 방어 — 미래 감염을 사전 대응.

    기존 대부분이 compromised 라벨에 '반응'하는 데 반해, 각 healthy 드론의 다음 스텝
    감염확률을 감염원 근접·SNR 열화로 추정해 '고위험 미감염' 노드에 선제 조치한다.
      P(감염_d) ∝ Σ_{c∈compromised} exp(-dist(d,c)/σ)  (+ 낮은 SNR 가중)
    고위험 → DeployDecoy(8) 유인/차단, 중위험 → BlockSuspicious(6), 감염 → 탈환/제거.
    액션 8을 실제로 쓰는 드문 아키텍처(액션 커버리지 분석 소재).
    """
    def __init__(self, n: int, sigma: float = 25.0, hi: float = 0.6, mid: float = 0.3):
        super().__init__(n)
        self.sigma = sigma; self.hi = hi; self.mid = mid

    def _risk(self, ctx):
        comp = ctx["compromised"]; pos = ctx.get("pos"); snr = ctx.get("snr")
        risk = [0.0]*self.n
        if pos is None or not comp:
            return risk
        for d in range(min(self.n, len(pos))):
            if d in comp:
                continue
            r = sum(math.exp(-_dist(pos, d, c)/self.sigma) for c in comp if c < len(pos))
            if snr is not None and snr[d] <= 6:
                r *= 1.5                         # 재밍당한 노드는 더 취약
            risk[d] = r
        return risk

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]
        risk = self._risk(ctx)
        mx = max(risk) or 1.0
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(4 if len(comp)/self.n < 0.5 else 3)
                continue
            rn = risk[d]/mx
            if rn >= self.hi:
                aids.append(6)                   # 고위험 → 선제 차단
            elif rn >= self.mid:
                aids.append(8)                   # 중위험 → 디코이 배치
            elif comp:
                aids.append(4)                   # 저위험이나 위협 존재 → 탈환 가담
            else:
                aids.append(1)
        return aids

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)


class StigmergyBlue(BlueBrainBase):
    """페로몬 격자 스티그머지 — 완전 분산 창발 분업 (개미군집형).

    격자(grid×grid 축소)에 위험 페로몬 필드를 유지: 감염 드론 위치에 침적, 매 스텝
    확산(이웃 평균)·증발(×decay). 각 드론은 자기 위치의 페로몬 농도만 보고 로컬 규칙으로
    행동한다(중앙 조정 없음). 중앙집중(graph) vs 분산 조정비용 비교 소재.
    """
    def __init__(self, n: int, cells: int = 20, decay: float = 0.85,
                 deposit: float = 1.0, hi: float = 0.5, mid: float = 0.15):
        super().__init__(n)
        self.cells = cells; self.decay = decay; self.deposit = deposit
        self.hi = hi; self.mid = mid
        self.field = None

    def _cell(self, ctx, d):
        pos = ctx.get("pos"); grid = 100.0
        if pos is None or d >= len(pos):
            return 0, 0
        cx = min(self.cells-1, int(pos[d][0]/grid*self.cells))
        cy = min(self.cells-1, int(pos[d][1]/grid*self.cells))
        return cx, cy

    def _update_field(self, ctx):
        import numpy as np
        if self.field is None:
            self.field = np.zeros((self.cells, self.cells))
        self.field *= self.decay                                  # 증발
        for c in ctx["compromised"]:                              # 침적
            cx, cy = self._cell(ctx, c)
            self.field[cx, cy] += self.deposit
        # 확산(간이 박스블러)
        f = self.field
        self.field = (f + np.roll(f,1,0) + np.roll(f,-1,0)
                      + np.roll(f,1,1) + np.roll(f,-1,1)) / 5.0

    def team_decide(self, ctx, agents):
        self._update_field(ctx)
        comp = ctx["compromised"]
        mx = float(self.field.max()) or 1.0
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            cx, cy = self._cell(ctx, d)
            conc = float(self.field[cx, cy])/mx
            if conc >= self.hi:
                aids.append(4)                   # 고농도 → 능동 탈환
            elif conc >= self.mid:
                aids.append(6)                   # 중농도 → 차단
            else:
                aids.append(1)                   # 저농도 → 감시(보존)
        return aids

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)


class MsgPassBlue(BlueBrainBase):
    """수제 GNN 메시지패싱 — 로컬 통신만으로 결정 (탈중앙).

    각 드론이 [감염플래그, snr_norm, gps_norm, link_up] 4D 피처를 k=2라운드 이웃과
    평균집계해 임베딩을 만들고, 고정 선형 스코어러로 action을 정한다. 학습 없이 손튜닝
    가중치. '전역 관측 vs 로컬 관측' ablation을 아키텍처로 구현(통신반경 축소 실험과 연결).
    """
    def __init__(self, n: int, rounds: int = 2):
        super().__init__(n)
        self.rounds = rounds

    def team_decide(self, ctx, agents):
        import numpy as np
        comp = ctx["compromised"]; pos = ctx.get("pos")
        snr = ctx.get("snr"); gps = ctx.get("gps_err"); link = ctx.get("link_up")
        n = self.n
        nbr = _proximity_graph(dict(ctx, n=n))
        # 초기 피처
        H = np.zeros((n, 4))
        for i in range(n):
            H[i,0] = 1.0 if i in comp else 0.0
            H[i,1] = (snr[i]/20.0) if snr is not None and i < len(snr) else 1.0
            H[i,2] = min(1.0, gps[i]/20.0) if gps is not None and i < len(gps) else 0.0
            H[i,3] = float(link[i]) if link is not None and i < len(link) else 1.0
        # 메시지패싱: 이웃 평균 집계
        for _ in range(self.rounds):
            H2 = H.copy()
            for i in range(n):
                if nbr.get(i):
                    H2[i] = 0.5*H[i] + 0.5*np.mean([H[j] for j in nbr[i]], axis=0)
            H = H2
        # 고정 스코어러: 이웃 감염압(H[:,0])이 크고 자기 취약(snr↓)하면 능동 대응
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                continue
            nb_inf = H[d,0]                       # 이웃까지 번진 감염 압력
            if nb_inf >= 0.25:
                aids.append(4)                   # 감염 압력 높음 → 탈환
            elif nb_inf >= 0.10:
                aids.append(6)                   # 중간 → 차단
            else:
                aids.append(1)
        return aids

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)


# ══════════════════════ Phase 4: EvoBlue (오프라인 최적화 반사 정책) ════════════

class EvoBlue(BlueBrainBase):
    """임계값 벡터 θ로 파라미터화된 반사 정책 — 오프라인 최적화로 성능 상한 추정.

    θ (8차원)로 per-drone 행동과 예산 하 우선순위를 정한다. run_evo.py가 CMA-ES/랜덤서치로
    θ를 최적화한 뒤 고정 배포(결정론). "graph/whittle를 넘으면 병목은 아키텍처가 아니라
    파라미터"라는 상한 추정.

    θ = [retake_r, block_r, snr_gate, hub_frac, prio_spread_w, prio_dist_w,
         prio_selfheal, conserve_r]  (모두 0~1 정규화, 내부에서 스케일)
    """
    DIM = 8
    DEFAULT = [0.6, 0.9, 0.3, 0.2, 1.0, 0.5, 0.5, 0.5]

    def __init__(self, n: int, theta=None):
        super().__init__(n)
        self.theta = list(theta) if theta is not None else list(self.DEFAULT)
        self._prio = {}

    def _p(self):
        t = self.theta; R = 60.0
        return dict(retake_r=t[0]*R, block_r=t[1]*R, snr_gate=t[2]*20.0,
                    hub_frac=t[3], spread_w=t[4], dist_w=t[5],
                    selfheal=t[6], conserve_r=t[7]*R)

    def team_decide(self, ctx, agents):
        comp = ctx["compromised"]; pos = ctx.get("pos"); snr = ctx.get("snr")
        p = self._p()
        nbr = _proximity_graph(dict(ctx, n=self.n))
        # 위협원 = 감염 + 강재밍
        threats = set(comp)
        if snr is not None:
            for i in range(self.n):
                if snr[i] <= 6:
                    threats.add(i)
        # 감염원 전파 잠재력
        spread_pot = {c: len(nbr.get(c, set()) - comp) for c in comp}
        self._prio = {}
        aids = []
        for a in agents:
            d = int(a.split("_")[-1])
            if d in comp:
                aids.append(3)
                self._prio[d] = p["selfheal"] * 2.0     # 자가치유 우선순위(튜너블)
                continue
            if not threats or pos is None or d >= len(pos):
                aids.append(1); continue
            # 최근접 위협 + 거리
            dmin, cnear = 1e9, None
            for j in threats:
                if j < len(pos):
                    dd = _dist(pos, d, j)
                    if dd < dmin:
                        dmin, cnear = dd, j
            if dmin <= p["retake_r"]:
                aids.append(4)
                sp = spread_pot.get(cnear, 0)
                self._prio[d] = p["spread_w"]*sp - p["dist_w"]*(dmin/60.0)
            elif dmin <= p["block_r"]:
                aids.append(6)
                self._prio[d] = p["spread_w"]*spread_pot.get(cnear, 0)*0.5
            elif dmin <= p["conserve_r"]:
                aids.append(2)                          # Analyse (약대응)
            else:
                aids.append(1)
        return aids

    def recovery_priority(self, ctx, agents, aids):
        return [self._prio.get(int(a.split("_")[-1]), 0.0) for a in agents]

    def step_decide(self, ctx):
        comp = ctx["compromised"]
        return 1 if not comp else (4 if len(comp)/self.n < 0.5 else 3)
