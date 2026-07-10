"""RAG-A → RAG-B end-to-end 통합 테스트.
실행: appendix/ 에서 `python -m attack_rag.integration_test`
필요: appendix/defense_rag/ (RAG-B) 가 같은 트리에 있어야 함.

시퀀스(아키텍처 §전체 흐름):
  관측(raw) → ① RAG-A(ATT&CK id + confidence + attack_class)
            → ② RAG-B(D3FEND 검색 → blue 행동 추천)
  RAG-A가 abstain(신뢰도 미달)이면 technique_id 없이 raw 관측만 넘겨
  RAG-B의 벡터 폴백 경로를 태운다(= 'ATT&CK 매칭 실패 시 raw fallback')."""
import os, sys, time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # LLM 훅 fork 경고 억제 (데모 화면 청결)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # appendix/ 를 path에

# 데모 녹화용 케이스 간 일시정지(초). 끄려면 DEMO_PAUSE=0
PAUSE = float(os.environ.get("DEMO_PAUSE", "2"))
from attack_rag.rag_a import RagA
from defense_rag.pipeline import DefenseRAG

print("[로드] RAG-A + RAG-B (all-MiniLM)...")
raga = RagA()                    # 관측 → ATT&CK
ragb = DefenseRAG()              # ATT&CK → CybORG 방어

# 케이스 구성: 아는 공격 1 + novel 2 (시나리오 23개·크로스워크 17신호 어디에도 없는 공격).
# novel 검증의 핵심 — RAG-A는 시나리오 목록이 아니라 ATT&CK/CAPEC 1456개 전체 KB를 검색한다.
cases = [
    ("[아는 공격: RF 재밍]",
     "sustained bandwidth flooding jamming on drone_07, SNR collapse, link down", ["drone_07"]),
    ("[★novel: 시나리오 밖 — DNS 터널링 C2]",
     "펌웨어 업데이트 이후 드론들이 주기적으로 미상 도메인에 DNS 질의를 보내고 응답 크기가 비정상적으로 큼", ["drone_09"]),
    ("[★novel: 적대적 패치 → abstain]",
     "카메라 영상에 특정 스티커 패턴이 잡힌 뒤 객체인식이 아군 차량을 지속적으로 오분류", ["drone_04"]),
]
ok = 0
for label, obs, host in cases:
    print(f"\n{'='*70}", flush=True)
    time.sleep(PAUSE)                       # 녹화용 호흡 — 구분선만 찍고 잠시 멈췄다 케이스 시작
    print(f"{label}\n[관측] {obs}")
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
    llm = " (LLM 선택)" if det.get("llm_used") else ""
    # direct_mapping이면 실제 매칭된 id 표시 (top-k 중 공식매핑 있는 첫 후보로 폴백될 수 있음)
    matched = det["technique_id"] if isinstance(det["technique_id"], str) else None
    path = det["path"] + (f"({matched})" if matched else "")
    print(f"  ② RAG-B → 경로={path} 추천 방어 {len(acts)}개:{llm}")
    for r in acts[:3]:
        basis = (f"유사도={r['score']}" if r.get("score") is not None
                 else "ATT&CK→D3FEND 공식매핑")
        print(f"       [{r['action_id']}] {r['action_name']} ← {r['d3fend_label']} "
              f"({r['d3fend_tactic']}, {basis})")
    if acts:
        print(f"  ③ 근거: {acts[0]['rationale']}")
    if acts:
        ok += 1
    else:
        print("       ⚠️ 추천 행동 없음 — 연결 실패")

if ok == len(cases):
    print(f"\n✅ end-to-end 통합 성공({ok}/{len(cases)}): 관측 → RAG-A(공격) → RAG-B(방어) → CybORG 행동")
else:
    sys.exit(f"\n❌ 통합 실패: {len(cases) - ok}/{len(cases)} 케이스에서 추천 행동이 비어 있음")
