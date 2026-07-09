# -*- coding: utf-8 -*-
"""★ HVT+RAG 통합 모델 (v2) — RAG 성능개선 실험의 최종 방어 정책.

구성: 방어정책(reach2/HVT 계열, 여기선 ReachV2 상속) + RAG 자세계층.
  - RAG-A(`src/attack_rag/rag_a.py`)가 관측 → attack_class(감염/비감염/재밍/unknown)를 산출
    (오프라인, `attack_rag/scenario_attack_class.json`로 주입 — 별도 env라 sim 인라인 호출 X).
  - 이 정책이 attack_class를 소비해 '대응 자세'만 라우팅(행동 직접결정 X, Posture Router).
실측(실전조건 채널①): naive 0.599 → 본 v2 0.931 (HVT 0.922 동률). `src/attack_rag/docs/` 참조.

--- 이하 자세 라우팅 로직 ---

이전 v1(naive): RAG-B의 top-1 권고행동을 그대로 적용 → 과소escalation(Monitor/Decoy) → 0.599.
v2(개선): RAG-A의 attack_class로 대응자세를 라우팅하되, 실제 탐지된 감염엔 항상
'봉쇄우선(containment-first)' 절차(NIST SP 800-61: 활성 감염은 관측이 아니라 격리→축출).

  compromise / non_compromise(tamper) / unknown → 봉쇄우선
        = reach2 컨테인먼트(탐지 감염 재장악[Isolate→Evict→Restore] + de-jam + relay)
  jamming → 복원력(resilience)
        = 실제 감염만 재장악, **FP는 파괴적 재장악 회피**(외부위협·evict할 호스트 없음, NIST)
          + de-jam/relay 로 가용성 유지

즉 자세만 다를 뿐, 라이브 탐지 증거가 있으면 언제나 봉쇄. RAG-B의 '관측만' 권고를
NIST 절차로 교정한 것. attack_class는 sim 정보(scenario_attack_class.json)에서 주입.
"""
from agents.reach2 import ReachV2


class RAGGuidedPolicy(ReachV2):
    name = "rag-guided"

    def __init__(self, attack_class="compromise"):
        self.attack_class = attack_class

    def step(self, comp, pos, env, live, ip2d, rng):
        if self.attack_class == "jamming":
            # 복원력 자세: 외부위협 → FP 파괴적 재장악 회피(fp=0), 실제 탐지 감염만 봉쇄 + de-jam
            saved = self.fp
            self.fp = 0.0
            try:
                return super().step(comp, pos, env, live, ip2d, rng)
            finally:
                self.fp = saved
        # compromise / non_compromise / unknown → 봉쇄우선(reach2 컨테인먼트)
        return super().step(comp, pos, env, live, ip2d, rng)
