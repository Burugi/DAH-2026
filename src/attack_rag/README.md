# attack_rag — 공격타입 판단 RAG (RAG-A)

드론 군집 사이버 공방에서 **공격 탐지(ATT&CK) → 방어 채택(D3FEND)** 파이프라인 중
**공격 탐지** 부분. 관측한 이상징후(SNR·GPS·세션·확산)만 보고 **"무슨 공격인지(ATT&CK 기법)"**
를 검색으로 식별한다. 특히 **처음 보는 공격(novel)** 대응이 목적.

`src/defense_rag/`(방어 채택 RAG-B)와 짝을 이룬다: **RAG-A(무슨 공격?) → RAG-B(어떻게 막나?)**.

> ★ **HVT+RAG 통합 모델(v2)은 `src/agents/rag_guided.py` (RAGGuidedPolicy)** — 방어정책(reach2/HVT) + RAG 자세계층.
> RAG-A가 attack_class를 산출하고, 그 정책이 자세를 라우팅한다(Posture Router). 실측·비교는 `docs/RAG_통합_검증_보고.md` §7.

## 파이프라인
```
관측 raw  →  크로스워크 매칭(관측→신호)  →  쿼리 확장  →  KB 벡터검색(1456 기법)
          →  출력 {기법ID, 신뢰도, target_host, attack_class}
```
- **메인:** 아는 징후 → 크로스워크(결정적 매칭) → 정확.
- **fallback:** 크로스워크 밖 novel → raw 설명 그대로 벡터검색.

## 출력 (개선 반영)
```python
from attack_rag.rag_a import RagA
r = RagA()                                   # 기본 all-MiniLM (프로덕션은 BAAI/bge-m3)
o = r.identify("SNR 급락 + 인접 순차 감염", target_host=[3, 5])
# o = {
#   "result": [{"id":"T1210","name":...,"confidence":0.55,"attack_class":"compromise","target_host":[3,5]}, ...],
#   "attack_class": "compromise",   # ★상위 라우팅 신호 (감염/비감염/재밍/unknown)
#   "abstain": False,               # ★신뢰도<0.35 → unknown(안전기본자세)
#   "matched_signal": "인접 순차 감염", "fallback": False
# }
```

### ★개선점 (naive 대비)
| 항목 | 개선 |
|---|---|
| **attack_class** | 감염/비감염/재밍/unknown 태그 — 방어 자세 라우팅 신호 |
| **aggregate 규칙** | NIST 봉쇄우선: top-3 어디든 감염 있으면 → compromise (혼합공격 포함) |
| **abstention** | 신뢰도<0.35 → unknown (억지 추측 대신 보수적 봉쇄) |

## 구성
| 파일 | 역할 |
|---|---|
| `rag_a.py` | 파이프라인 (관측→공격유형+attack_class) |
| `embed_validate.py` | held-out 검증 (`python embed_validate.py BAAI/bge-m3`) |
| `integration_test.py` | RAG-A→RAG-B end-to-end (defense_rag 필요) |
| `attack_capec_kb.json` | 공격 지식DB (ATT&CK Ent/ICS/Mobile + CAPEC = 1456 기법, v18.1) |
| `drone_crosswalk.json` | 관측→공격 크로스워크 17신호 (핵심) |
| `response_procedures.json` | NIST/D3FEND 정식 대응절차 (attack_class별) |
| `scenario_attack_class.json` | 시나리오별 attack_class (참조) |
| `heldout_procedure_test.json` | novel 검증셋 (16k procedure examples 샘플) |
| `docs/` | 구축 문서·통합 검증 보고·아키텍처 다이어그램 |

## 실행 (별도 env 권장 — CybORG sim과 충돌 방지)
```bash
pip install -r attack_rag/requirements.txt   # sentence-transformers 등
cd src
python -m attack_rag.rag_a                    # 데모
python attack_rag/embed_validate.py BAAI/bge-m3   # 검증
```
⚠️ **sentence-transformers가 torch/numpy를 끌어와 CybORG(numpy==1.23) 환경과 충돌** → 별도 env에 설치.

## 인터페이스 (RAG-B와 합의)
| 항목 | 합의 |
|---|---|
| B1 출력형식 | `{기법ID, 설명, 신뢰도, target_host}` (+ attack_class) |
| B2 ATT&CK 버전 | **v18.1** (RAG-A KB·시나리오태그·D3FEND 3자 통일) |
| B3 fallback | ID 미확정 시 raw 설명 → RAG-B 벡터검색 |

## 데이터 출처 (공개)
MITRE ATT&CK v18.1 (Enterprise/ICS/Mobile), CAPEC, MITRE D3FEND, NIST SP 800-61 — 모두 공개표준.
원시 STIX/XML 덤프는 용량 문제로 미포함(재다운로드 가능); 파생 KB만 포함.
