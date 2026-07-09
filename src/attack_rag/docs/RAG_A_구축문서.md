# RAG-A 구축 문서 — 공격타입 판단 (이준영)

> **한 줄:** 드론이 관측한 이상징후(SNR·GPS·세션·확산)만 보고 **"무슨 공격인지(ATT&CK 기법)"** 를 검색으로 알아내는 시스템. 특히 **처음 보는 공격(novel)** 대응이 목적.

---

## 1. RAG-A가 뭐고 왜 필요한가 (쉬운 설명)

우리 HVT 방어는 웜을 **억제**는 잘하지만 공격이 **무슨 종류인지**는 모릅니다. RAG-A가 그걸 알려줍니다:

```
관측: "3번 드론 SNR 급락 + 통신 두절"   →   RAG-A   →   "이건 재밍(T1498/T1464) 공격"
```

**왜 "검색(RAG)"이냐:**
- 공격 종류를 규칙(if-else)으로 다 짜면 → **처음 보는 공격**은 못 잡음.
- 대신 **공격 지식DB(ATT&CK 등)를 벡터검색** → 새 공격도 "가장 비슷한 알려진 공격"으로 매칭 = **일반화.**

**두 RAG 협업:**
```
공격 징후 → [RAG-A: 무슨 공격? (내 담당)] → 공격유형 → [RAG-B: 어떻게 막나? (협업)] → 방어책
```

---

## 2. 전체 구조 (파이프라인)

```
① 관측 (sim)          SNR·GPS_err·link_up·red_session·확산패턴
        ↓
② 크로스워크 매칭      관측 → 어느 "신호"인지 (SNR↓=재밍 신호 등)
        ↓
③ 쿼리 생성           신호의 ATT&CK 검색어(+ novel이면 raw 설명)
        ↓
④ 벡터검색            공격 지식DB(1456 기법)에서 top-k 검색
        ↓
⑤ 출력 (B1)          [{기법ID, 이름, 신뢰도, target_host}]
```

**메인 경로 vs fallback:**
- **메인:** 구조화 관측 → 크로스워크(결정적 매칭) → 검색. 아는 징후는 정확.
- **fallback (B3):** 크로스워크에 없는 **완전 novel** → raw 설명을 그대로 벡터검색.

---

## 3. 데이터 구성 (각각 뭔지)

`rag_data/` 폴더:

| 파일 | 뭔지 | 크기 |
|---|---|---|
| **attack_capec_kb.json** | **공격 지식DB.** ATT&CK Enterprise(691)+ICS(83)+Mobile(124)+CAPEC(558) = **1456 기법**. 각 기법의 이름·설명·탐지·전술 | 검색 대상 |
| **drone_crosswalk.json** | **관측→공격 매핑표(핵심).** 17신호(SNR↓→T1498 등), OSI 계층별. ATT&CK엔 없는 "징후→공격" 연결을 우리가 만든 것 | RAG-A의 뇌 |
| **heldout_procedure_test.json** | **검증셋.** 실제 위협그룹이 기법 쓴 서술 16k개(라벨 있음). novel 표현 일반화 측정용 | 검증 |
| enterprise/ics/mobile-attack-18.1.json | ATT&CK 원본 STIX (v18.1) | 아카이브 |
| capec_latest.xml | CAPEC 원본 | 아카이브 |
| attack_v18.1_tfidf.pkl | TF-IDF 벡터 인덱스(프로토타입) | 인덱스 |

**★왜 KB를 4개 도메인 합쳤나:**
- Enterprise만으론 **드론 공격(재밍·GPS·제어조작)** 이 약함.
- **Mobile**(T1464 GPS/RF 재밍) + **ICS**(T0831 제어조작·T0855 무단명령) = 드론 cyber-physical 커버.
- **CAPEC** = 처음 보는 변종을 상위 공격패턴으로 폴백.

**★크로스워크가 왜 핵심:**
- ATT&CK은 "공격이 뭔지"만 서술 → **"SNR 3이면 T1498"은 안 적혀 있음.**
- 우리 입력은 "관측"이라 → **관측을 ATT&CK 검색어로 번역하는 사전**이 있어야 검색이 맞물림. 이게 크로스워크.

---

## 4. 코드 (`rag_a.py` — 어떻게 동작하나)

```python
from rag_a import RagA
r = RagA(model="BAAI/bge-m3")          # 임베딩 모델 로드 + KB 임베딩

result = r.identify(
    obs_text="드론 3번 SNR 급락, 통신 두절, 인접 드론도 하락",   # 관측
    target_host=[3, 5]                                          # 공격받는 드론
)
# → {'result': [{'id':'T1498','name':'Network Denial of Service',
#                'confidence':0.72,'target_host':[3,5],'domain':'enterprise'}, ...],
#    'matched_signal':'SNR 광역 하락', 'fallback':False}
```

**내부 동작:**
1. 관측을 임베딩 → 크로스워크 17신호 중 가장 가까운 것 매칭(유사도>0.35).
2. 매칭되면 그 신호의 ATT&CK 검색어 결합(정확도↑), 아니면 raw만(fallback).
3. KB(1456) 벡터검색 → top-k.
4. **B1 출력:** `{기법ID, 이름, 신뢰도, target_host}` + 매칭신호 + fallback 여부.

---

## 5. 검증 (방법 + 결과)

**"novel 공격 어떻게 테스트?"** — 4단계 검증 체계:

| 단계 | 방법 | 결과 |
|---|---|---|
| 1. 알려진(23시나리오·크로스워크) | ground truth 대조 | ✅ 크로스워크 14/17 |
| 2. **novel 표현**(16k procedure examples) | 다른 표현으로 같은 기법 검색되나 | **TF-IDF 0.29 → 의미 0.53** |
| 3. 드론 도메인 novel | UAV 특화 공격 큐레이션 | (예정) |
| 4. 무라벨 완전 novel | fallback + 신뢰도 임계 | 설계됨 |

**핵심 결과 — 의미검색이 어휘검색 압도:**
| recall@5 | TF-IDF(어휘) | all-MiniLM(의미) |
|---|:--:|:--:|
| novel procedure | 0.294 | **0.532** (1.8배) |

→ **"처음 보는 표현이라도 공격을 식별"** = RAG-A 목적 달성 가능성 정량 입증.

---

## 6. GPU 환경 세팅 가이드 (BGE-M3 프로덕션)

**왜 GPU:** BGE-M3(568M)는 CPU서 너무 느림/메모리 초과. GPU(단일이면 충분)서 실행.

### 세팅 단계
```bash
# 1) Python 환경 (3.10+)
conda create -n raga python=3.10 && conda activate raga

# 2) GPU torch 설치 (CUDA 버전 맞춰 — 예: CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3) 임베딩 라이브러리
pip install sentence-transformers          # dense 검색용
pip install FlagEmbedding                   # ★하이브리드(dense+sparse) 원할 때

# 4) 데이터 복사 (이 rag_data/ 폴더 + rag_a.py + embed_validate.py 를 GPU 서버로)

# 5) 실행 — held-out 검증 (BGE-M3 성능 확인)
python embed_validate.py BAAI/bge-m3
#   → recall@k 출력 (all-MiniLM 0.53 대비 상회 예상). 모델 최초 실행시 ~2.3GB 자동 다운로드.

# 6) RAG-A 사용
python rag_a.py                             # 데모 (rag_a.py의 model= 을 BAAI/bge-m3 로)
```

### ⚠️ 환경 gotcha (실측 확인)
- **Python 3.11~3.12 필수** — 3.13/3.14는 PyTorch CUDA 휠 미출시(CPU만). uv면 `uv venv --python 3.11`.
- **torch ≥ 2.6 필요** (transformers 보안 요구) → **cu124 인덱스** 사용(`--index-url .../cu124`). cu121은 torch 2.5까지라 부족.
- **8GB VRAM(RTX 4060)**: BGE-M3 대규모 배치는 **segfault** → `m.max_seq_length=256` + `batch_size=4~8` + 배치마다 `torch.cuda.empty_cache()`.
- **KB 임베딩은 1회성** — 한 번 임베딩해 저장(np.save)하면 서빙은 쿼리만 임베딩(빠름). 그러니 임베딩 속도는 큰 문제 아님(CPU 13분도 OK).

### 실측 결과 (이 환경서 확인)
- CUDA 작동: torch 2.6.0+cu124, RTX 4060 인식 ✅
- BGE-M3 다국어 확인: "network denial of service jamming"(EN) ↔ "재밍 공격 신호 급락"(KO) 유사도 **0.614** → **한국어 관측→영어 ATT&CK 연결 가능**(all-MiniLM은 불가). 이게 BGE-M3 채택의 결정적 근거.
- BGE-M3 dense held-out recall@5 = 0.513 (경량 all-MiniLM 0.532와 유사 — 영어-dense선 차이 작음). **하이브리드(아래) + 한국어에서 BGE-M3 우위.**

### ★하이브리드 검색 (연구 권장 — 정확도 최상)
`sentence-transformers`는 **dense(의미)만**. 기법ID(T1464)·전문용어 **exact match**까지 하려면 **FlagEmbedding의 BGEM3FlagModel**로 dense+sparse 동시:
```python
from FlagEmbedding import BGEM3FlagModel
m = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)   # GPU
out = m.encode(texts, return_dense=True, return_sparse=True)
# dense(의미) + sparse(어휘 exact) 점수를 가중합 → 최종 랭킹
```

### 한국어 질의 중심이면
`BAAI/bge-m3` 자리에 **`nlpai-lab/KURE-v1`** (고려대, bge-m3 한국어 파인튜닝, 아키텍처 동일 → 코드 그대로). MTEB-ko 최상위.

---

## 7. 남은 작업 / 한계

- **BGE-M3 프로덕션 임베딩** — GPU 환경서 §6 실행 (현재 CPU 프로토타입은 all-MiniLM).
- **드론 도메인 검증셋(3단계)** — UAV 특화 공격 큐레이션 추가.
- **하이브리드 검색** — FlagEmbedding으로 dense+sparse (기법ID exact match).
- **크로스워크 확장** — sim 관측 지표 추가 시 크로스워크 신호 추가.
- **한계:** ATT&CK v18.1 고정(방어 RAG-B와 3자 통일). target_host는 웜/점령은 특정 드론, 재밍은 광역(목록/area).

---

## 8. 협업 인터페이스 (방어 RAG-B와 합의됨)

| 항목 | 합의 |
|---|---|
| B1 출력형식 | `{기법ID, 설명, 신뢰도, target_host}` (top-k) |
| B2 ATT&CK 버전 | **v18.1** (3자 통일: RAG-A KB·시나리오태그·D3FEND) |
| B3 fallback | ID 미확정 시 raw 설명 → RAG-B 벡터검색 |
| C1 출력 | RAG-B 방어책 → CybORG A5 행동 |

---

*생성물: `rag_data/` (KB·크로스워크·검증셋·인덱스), `rag_a.py` (파이프라인), `embed_validate.py` (검증), `RAG_A_데이터_임베딩_평가.md` (리서치).*
