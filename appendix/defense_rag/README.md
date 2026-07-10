# defense_rag — D3FEND 기반 방어 채택 RAG

드론 군집 사이버 공방(CybORG CC3)에서 **공격 탐지(ATT&CK) → 방어 채택(D3FEND)**
파이프라인 중 **방어 채택** 부분. 처음 보는 공격에도 외부 지식(MITRE D3FEND)을
검색해 CybORG blue 행동을 추천한다.

전체 배경·설계 판단은 팀 공유 문서 `DEFENSE_RAG_PLAN.md` 참고(저장소 외부).
RAG-A와의 end-to-end 통합·실측은 `../attack_rag/docs/RAG_통합_검증_보고.md` 참고.

## 하이브리드 2경로
```
탐지 입력(ATT&CK id + 관측 텍스트)
  ├─ ① id 확정 & 공식 매핑 있음 → AttackLookup 직접 조회      (정확)
  └─ ② id 미확정 / 매핑 없음(처음 보는 공격) → 벡터 검색       (폴백)
        → D3FEND 기법 → action_map → CybORG blue 행동(0-9)
```
설계 핵심: 시나리오 ATT&CK id의 **약 절반만** D3FEND 직접 매핑이 있다
(DoS·데이터조작류는 없음). 이 공백을 ②번 벡터 검색이 메운다 = 하이브리드의 존재 이유.

## 구성
| 파일 | 역할 | Task |
|---|---|---|
| `data/download_data.sh` | D3FEND 공식 덤프 다운로드 (원본, 미커밋) | A1 |
| `rag_data/` | 파생 산출물·임베딩 인덱스 전용 폴더 (커밋됨) | — |
| `build_kb.py` | 덤프 파싱 → 기법 271개 + ATT&CK 매핑 325개 | A1·A2 |
| `index.py` | LangChain 청킹 → 임베딩 → numpy 벡터 인덱스 + 검색 | A3 |
| `lookup.py` | ATT&CK id → D3FEND 기법 직접 조회 | A4 |
| `action_map.py` | D3FEND 기법 → CybORG blue 행동 매핑표(초안) | A5 |
| `pipeline.py` | 탐지 입력 → 추천 행동 e2e | A6 |
| `verify_c2.py` | C2 역방향 검증 (기법→행동 전수, 죽은행동) | C2 |
| `llm_hook.py` | 후보 중 LLM 선택 + 근거 (오프라인 폴백) | §9 |

데이터는 두 폴더로 분리 관리한다(attack_rag와 동일 컨벤션):
- `data/` — D3FEND 원본 덤프(~50MB, 미커밋, `download_data.sh`로 재생성)
- `rag_data/` — `d3fend_techniques.jsonl` · `attack_to_d3fend.json` · `d3fend_index.npz`

## 실행 (팀원용 — 바로 돌리기)
파생 산출물(`rag_data/`)이 **저장소에 포함**돼 있어 다운로드·재빌드 없이 바로 실행된다.
```bash
pip install -r defense_rag/requirements.txt   # sentence-transformers 등 (별도 env 권장)
cd appendix
python -m defense_rag.pipeline    # mock 입력 e2e 데모
python -m defense_rag.verify_c2   # 시나리오 23개 매핑 커버리지 검증
```
파이썬에서 직접 호출:
```python
from defense_rag.pipeline import DefenseRAG
rag = DefenseRAG()
out = rag.recommend({"detections": [{"technique_id": "T1210", "observation": "...",
                                      "target": {"type": "hosts", "value": ["drone_07"]}}]})
```

## 원본 재생성 (필요할 때만)
KB를 처음부터 다시 만들 때만 필요(원본 덤프 ~50MB, 저장소엔 미포함):
```bash
bash defense_rag/data/download_data.sh   # 공식 덤프 다운로드
python -m defense_rag.build_kb           # 파싱 → rag_data/ jsonl + json
python -m defense_rag.index              # LangChain 청킹 + 벡터 인덱스 재생성 (rag_data/*.npz)
```

## 임베딩·청킹
`sentence-transformers` 로컬 모델 `all-MiniLM-L6-v2`(384d). 최초 1회만 모델
다운로드 후 완전 오프라인. API 키 불필요. 시뮬레이터(numpy 1.23 고정 py3.11)와
**별도 프로세스**로 도므로 의존성 충돌 없음.

인덱스 빌드 시 기법 문서를 LangChain `RecursiveCharacterTextSplitter`
(chunk_size=1000자, overlap=100자 — 임베딩 모델 윈도 기준, attack_rag와 동일)로
청크 분할해 임베딩한다. 현재 KB는 전 문서가 윈도 안이라 1문서=1청크(잘림 없음)이고,
윈도 초과 문서만 자동 분할된다. 검색은 청크 단위 유사도를 기법 단위 max로 집계해
top-k 기법을 반환한다. 청킹 의존성은 빌드 시에만 필요하고 검색은 npz만 읽는다.
※ attack_rag heldout 실측: 윈도보다 잘게(400자) 쪼개면 recall@5 0.533→0.479 하락.

## LLM 훅
`llm_hook.py`는 `ANTHROPIC_API_KEY`가 있으면 Claude(기본 `claude-sonnet-5`,
env `DEFENSE_RAG_LLM_MODEL`로 변경)가 후보 방어 행동 중 하나를 골라 근거를 쓴다.
구조화 출력으로 `{action_id, rationale}`만 받고, 후보 밖 id·오류·키 없음이면
규칙 기반으로 폴백(오프라인 시연 보장). 출력의 `llm_used`로 어느 경로였는지 표시.

## 남은 일 (합의 필요, PLAN §5-B/C)
- D3FEND의 IT 편향으로 드론/GPS/DoS 관측은 벡터 점수가 낮음(0.2~0.46) → 도메인 힌트 보강(옵션)

완료된 항목:
- ~~LLM 훅 라이브 호출 검증~~ ✅ 2026-07-10 실검증 (claude-sonnet-5) — integration_test
  3케이스 전부 후보 내 선택 + 한국어 근거 생성 확인. 적대적 패치 케이스에선 규칙 순서
  (Monitor 우선)를 LLM이 오작동 위험 판단으로 Failsafe 우선으로 재정렬 = 과소대응 교정.
- ~~B4 통합 테스트~~ ✅ RAG-A 실제 output 연결 완료 — `attack_rag/integration_test.py`
  3/3 통과 (abstain → raw 관측 벡터 폴백 포함)
- ~~ATLAS 노드 KB 포함 확인~~ ✅ 확인 결과 KB(271기법)·매핑에 ATLAS(AML.*) 없음.
  단 A11 시나리오는 일반 Enterprise T-code(T1562/T1565/T1036)만 쓰고 셋 다
  직접매핑이 없어 벡터 폴백으로 처리됨 — 하이브리드 설계상 의도된 경로.
