"""RAG-A → RAG-B end-to-end 통합 테스트.
실행: appendix/ 에서 `python -m attack_rag.integration_test`
필요: appendix/defense_rag/ (RAG-B) 가 같은 트리에 있어야 함.

시퀀스(아키텍처 §전체 흐름):
  관측(raw) → ① RAG-A(ATT&CK id + confidence + attack_class)
            → ② RAG-B(D3FEND 검색 → blue 행동 추천)
  RAG-A가 abstain(신뢰도 미달)이면 technique_id 없이 raw 관측만 넘겨
  RAG-B의 벡터 폴백 경로를 태운다(= 'ATT&CK 매칭 실패 시 raw fallback')."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # appendix/ 를 path에
from attack_rag.rag_a import RagA
from defense_rag.pipeline import DefenseRAG

print("[로드] RAG-A + RAG-B (all-MiniLM)...")
raga = RagA()                    # 관측 → ATT&CK
ragb = DefenseRAG()              # ATT&CK → CybORG 방어

cases = [
    ("sustained bandwidth flooding jamming on drone_07, SNR collapse, link down", ["drone_07"]),
    ("neighbor drones sequentially infected, worm lateral spread", ["drone_02", "drone_04"]),
    ("정체불명의 미세한 텔레메트리 지연 발생", ["drone_01"]),          # 저신뢰 → abstain → raw 폴백 경로
]
ok = 0
for obs, host in cases:
    print(f"\n{'='*70}\n[관측] {obs}")
    # ① RAG-A: 관측 → 공격타입 (id + confidence + attack_class)
    a = raga.identify(obs, target_host=host, topk=3)
    top = a["result"][0]
    print(f"  ① RAG-A → {top['id']} ({top['name']}) conf={top['confidence']} "
          f"class={a['attack_class']} abstain={a['abstain']} [매칭:{a['matched_signal']}]")
    # ② RAG-B: 공격타입 → 방어. abstain이면 id 대신 raw 관측으로 폴백(§6 스키마).
    tid = None if a["abstain"] else [r["id"] for r in a["result"]]
    b = ragb.recommend({"detections": [{
        "technique_id": tid, "confidence": top["confidence"],
        "observation": obs, "target": {"type": "hosts", "value": host}}]})
    det = b["results"][0]
    acts = det["recommended_actions"]
    print(f"  ② RAG-B → 경로={det['path']} 추천 방어 {len(acts)}개:")
    for r in acts[:3]:
        print(f"       [{r['action_id']}] {r['action_name']} ← {r['d3fend_label']} "
              f"({r['d3fend_tactic']}) score={r['score']}")
    if acts:
        ok += 1
    else:
        print("       ⚠️ 추천 행동 없음 — 연결 실패")

if ok == len(cases):
    print(f"\n✅ end-to-end 통합 성공({ok}/{len(cases)}): 관측 → RAG-A(공격) → RAG-B(방어) → CybORG 행동")
else:
    sys.exit(f"\n❌ 통합 실패: {len(cases) - ok}/{len(cases)} 케이스에서 추천 행동이 비어 있음")
