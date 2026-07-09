"""RAG-A → RAG-B end-to-end 통합 테스트.
실행: src/ 에서 `python -m attack_rag.integration_test`
필요: src/defense_rag/ (RAG-B, defense-rag 브랜치) 가 병합돼 있어야 함."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ 를 path에
from attack_rag.rag_a import RagA
from defense_rag.pipeline import DefenseRAG

print("[로드] RAG-A + RAG-B (all-MiniLM)...")
raga = RagA()                    # 관측 → ATT&CK
ragb = DefenseRAG()              # ATT&CK → CybORG 방어

cases = [
    ("sustained bandwidth flooding jamming on drone_07, SNR collapse, link down", ["drone_07"]),
    ("neighbor drones sequentially infected, worm lateral spread", ["drone_02","drone_04"]),
]
for obs, host in cases:
    print(f"\n{'='*70}\n[관측] {obs}")
    # ① RAG-A: 관측 → 공격타입
    a = raga.identify(obs, target_host=host, topk=3)
    tid = a['result'][0]['id']; tname = a['result'][0]['name']
    print(f"  ① RAG-A → {tid} ({tname}) conf={a['result'][0]['confidence']} [매칭:{a['matched_signal']}]")
    # ② RAG-B: 공격타입 → 방어 (B1 인터페이스)
    b = ragb.recommend({"detections":[{"technique_id": tid, "observation": obs,
                                       "target":{"type":"hosts","value":host}}]})
    det = b["detections"][0] if isinstance(b,dict) and "detections" in b else b
    acts = det.get("recommendations") or det.get("actions") or det.get("candidates") or []
    print(f"  ② RAG-B → 추천 방어 (기법 {tid} 기반):")
    for r in acts[:3]:
        print(f"       [{r.get('action_id')}] {r.get('action_name')} ← {r.get('d3fend_label')} ({r.get('d3fend_tactic')}) score={r.get('score')}")
print("\n✅ end-to-end 통합 성공: 관측 → RAG-A(공격) → RAG-B(방어) → CybORG 행동")
