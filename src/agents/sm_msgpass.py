"""[구조 파악용] 모델: msgpass   (로컬 GNN 메시지패싱(분산))
레지스트리 정의: MsgPassBlue(n)
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


# ── 공용 헬퍼(원본 experimental.py 모듈 레벨) ──
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
