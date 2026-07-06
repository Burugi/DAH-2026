# defense_rag — D3FEND 기반 방어 채택 RAG

드론 군집 사이버 공방(CybORG CC3)에서 **공격 탐지(ATT&CK) → 방어 채택(D3FEND)**
파이프라인 중 **방어 채택** 부분. 처음 보는 공격에도 외부 지식(MITRE D3FEND)을
검색해 CybORG blue 행동을 추천한다.

전체 배경·설계 판단은 저장소 루트의 `DEFENSE_RAG_PLAN.md` 참고.

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
| `data/download_data.sh` | D3FEND 공식 덤프 다운로드 | A1 |
| `build_kb.py` | 덤프 파싱 → 기법 271개 + ATT&CK 매핑 325개 | A1·A2 |
| `index.py` | 기법 임베딩 → numpy 벡터 인덱스 + 검색 | A3 |
| `lookup.py` | ATT&CK id → D3FEND 기법 직접 조회 | A4 |
| `action_map.py` | D3FEND 기법 → CybORG blue 행동 매핑표(초안) | A5 |
| `pipeline.py` | 탐지 입력 → 추천 행동 e2e | A6 |

## 실행
```bash
cd src
bash defense_rag/data/download_data.sh   # 최초 1회 (원본 덤프 ~50MB)
python -m defense_rag.build_kb           # 파싱 → jsonl + json
python -m defense_rag.index              # 벡터 인덱스 생성 (.npz)
python -m defense_rag.pipeline           # mock 입력 e2e 데모
```

## 임베딩
`sentence-transformers` 로컬 모델 `all-MiniLM-L6-v2`(384d). 최초 1회만 모델
다운로드 후 완전 오프라인. API 키 불필요. 시뮬레이터(numpy 1.23 고정 py3.11)와
**별도 프로세스**로 도므로 의존성 충돌 없음.

## 남은 일 (합의 필요, PLAN §5-B/C)
- B1 입력 인터페이스 확정(탐지 담당자) — 현재는 §6 초안 형식 사용
- C2 `action_map.py` 초안을 실제 CybORG 행동 공간으로 검증
- LLM 근거 생성 훅(현재 규칙 기반 폴백) — Claude API 연결 시 교체
- D3FEND의 IT 편향으로 드론/GPS/DoS 관측은 벡터 점수가 낮음 → 도메인 힌트 보강 검토
