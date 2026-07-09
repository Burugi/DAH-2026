# RAG-A (공격타입 판단) — 데이터 소스 & 임베딩 모델 평가

> 목적: sim 관측 징후(SNR↓·GPS 이상·비정상 세션·확산) → **무슨 공격인지(ATT&CK 기법 ID)** 식별, 특히 **novel(처음 보는) 공격** 대응.
> 도메인: 드론 스웜 / MANET / cyber-physical(OT). 현재 데이터: **MITRE ATT&CK Enterprise v18.1 (691 기법)**. 협업 RAG-B는 D3FEND v1.4.0.
> 조사 방식: 공개 자료 딥리서치(WebSearch/WebFetch), 다중 소스 교차확인. 불확실 항목은 "(추정)" 표기.
> 작성일: 2026-07-07

---

## 0. 핵심 요약 (TL;DR)

1. **현 ATT&CK Enterprise만으로는 불충분.** Enterprise는 IT 측(초기침투·C2·유출·펌웨어 변조)은 잘 커버하나, 드론의 핵심 공격군인 **RF 재밍·GPS 스푸핑·물리계층/무선** 은 매핑이 약함. 이들은 ATT&CK **Mobile**(T1464 Network DoS = radio/GPS 재밍)과 **ICS**(cyber-physical 영향)에, 그리고 드론 전용 학술 분류에 더 잘 들어맞음.
2. **추천 데이터 조합**: `ATT&CK Enterprise(백본) + ATT&CK ICS + ATT&CK Mobile` 3개 도메인을 통합하고, **CAPEC**(공격패턴 추상화 → novel 일반화)와 **드론/UAV 학술 위협 분류(OSI 계층 taxonomy)** 를 보완 소스로 결합. Sigma는 "관측→ATT&CK" 브리지로 선택적 활용.
3. **추천 임베딩 모델**: 1순위 **BGE-M3**(다국어 EN+KO, MIT, 8K 컨텍스트, dense+sparse 하이브리드 → 기법ID/키워드 exact match 유리). 한국어 쿼리 비중이 크면 **KURE-v1**(bge-m3 한국어 파인튜닝, drop-in 교체). 경량 대안 **multilingual-e5-large** 또는 **gte-multilingual-base**.
4. **핵심 발견**: 드론 sim 관측치(SNR↓·GPS lock 상실·패킷 드롭 버스트)를 공격으로 잇는 **"관측→공격" 매핑은 ATT&CK 본문엔 거의 없다.** 이 매핑을 직접 제공하는 것은 (a) 드론 학술 taxonomy(관측 anomaly ↔ attack 표), (b) Sigma 룰(로그 시그니처 → ATT&CK 태그) 두 가지. RAG-A 검색 품질의 병목은 임베딩 모델보다 **이 관측→공격 크로스워크 데이터의 확보**.

---

## 1. 리서치 ① — 데이터 소스 평가 + 대안

### 1.1 소스별 비교표

평가축: 신뢰도(권위·큐레이션) / 범용성(도메인 폭) / **드론 커버리지**(RF·GPS·물리) / 정확도(설명 품질) / 최신성 / 라이선스 / 기계판독성. (5점 척도, ★=1)

| 소스 | 신뢰도 | 범용성 | 드론 커버리지 | 정확도(설명) | 최신성 | 라이선스 | 기계판독성 | RAG-A 적합성 |
|---|---|---|---|---|---|---|---|---|
| **ATT&CK Enterprise v18** (현재) | ★★★★★ | ★★★★☆ IT 중심 | ★★☆☆☆ 약함 | ★★★★★ | ★★★★★ | MITRE ToU (무료·상용가능) | STIX 2.1/JSON | **백본** |
| **ATT&CK ICS** (~79 기법, 12 tactic) | ★★★★★ | ★★★☆☆ OT | ★★★★☆ cyber-physical 영향 | ★★★★★ | ★★★★☆ | MITRE ToU | STIX/JSON | **필수 보완** |
| **ATT&CK Mobile** (T1464 등) | ★★★★★ | ★★★☆☆ 모바일/RF | ★★★★☆ radio/GPS 재밍 | ★★★★☆ | ★★★★☆ | MITRE ToU | STIX/JSON | **필수 보완(RF)** |
| **CAPEC v3.9** (~559 패턴) | ★★★★★ | ★★★★★ 공격패턴 전반 | ★★★☆☆ 추상적 | ★★★★☆ | ★★★☆☆ | MITRE ToU | XML/CSV | **novel 일반화에 유용** |
| **CWE / CVE** | ★★★★★ | ★★★★☆ 취약점 | ★★☆☆☆ 간접 | ★★★☆☆ | ★★★★★ | 공개 | JSON/XML | 낮음(취약점 중심, 관측 아님) |
| **Sigma rules** (SigmaHQ, 수천 룰) | ★★★★☆ | ★★★☆☆ IT 로그 | ★★☆☆☆ IT 로그 편향 | ★★★★☆ | ★★★★★ | DRL 1.1 (상용가능·저자표기) | YAML + ATT&CK 태그 | **관측→ATT&CK 브리지** |
| **드론/UAV 학술 taxonomy** (OSI 계층) | ★★★☆☆ 논문별 상이 | ★★★☆☆ 드론 특화 | ★★★★★ RF·GPS·펌웨어 | ★★★★☆ | ★★★★☆ | 논문 라이선스 상이 | 표/텍스트(수작업 정형화 필요) | **관측→공격 매핑 핵심** |
| **CTI 피드**(실 위협 인텔) | ★★★☆☆~★★★★☆ | ★★★★☆ | ★★★☆☆ | 가변 | ★★★★★ | 상용/공개 혼재 | STIX/TAXII | novel 최신성 보완(운영 부담↑) |

### 1.2 각 소스 평가 상세

**ATT&CK Enterprise (현재 데이터)**
- 강점: 최고 권위·큐레이션, 691 기법의 서술 품질 우수, STIX 2.1 기계판독, D3FEND(RAG-B)와 직접 연결(mitigations/relationships). Novel 대응 시 유사 기법 서술 검색에 좋은 백본.
- 약점: **IT(Windows/Mac/Linux/Cloud) 행위 중심.** 드론 핵심인 **GPS 스푸핑·RF 재밍·물리계층 신호 교란**은 Enterprise 매트릭스에 대응 기법이 빈약. 펌웨어류는 일부 존재(T1542 Pre-OS Boot, T1495 Firmware Corruption)하나 무선/항법 도메인은 공백.
- 판단: **필요조건이나 충분조건 아님.** 백본으로 유지하되 반드시 확장.

**ATT&CK ICS** — 드론 = cyber-physical system(CPS). ICS 매트릭스는 IT→OT 전이(Wireless Compromise, Internet Accessible Device 등 initial access)와 **물리적 영향**(Loss of Control, Loss of Safety, Damage to Property, Denial of Control)을 포착. 액추에이터 하이재킹·제어 상실 등 드론 CPS 시나리오와 정합. → **통합 권장.**

**ATT&CK Mobile** — **T1464 Network Denial of Service**가 "adversary가 Wi-Fi/cellular/**GPS** 신호를 재밍하여 장치 통신을 차단"을 명시적으로 기술. 드론의 재밍/RF DoS/무선 위협 매핑에 가장 근접한 ATT&CK 도메인. → **통합 권장(RF 계층).**

**CAPEC** — 공격 "패턴" 추상화(약 559개, v3.9). CWE↔CAPEC↔ATT&CK 공식 크로스워크 제공. 개별 기법보다 상위 패턴이라 **처음 보는 변종을 상위 패턴으로 일반화**하는 데 유용(novel 대응 강점). 서술은 다소 추상적. → 보완 소스로 결합.

**CWE/CVE** — 취약점·약점 중심. "관측 징후→공격 유형" 질문형과 결이 다름(무엇이 취약한가 vs 무슨 공격인가). 특정 드론 펌웨어/스택 CVE 참조엔 유용하나 RAG-A 1차 소스로는 부적합. → 낮은 우선순위.

**Sigma rules** — YAML 탐지 시그니처, 각 룰에 ATT&CK 기법 태그. **"로그/이벤트 관측 → ATT&CK 기법"** 매핑을 직접 담고 있어 RAG-A의 "관측→공격" 구조와 가장 유사. 라이선스 DRL 1.1(상용 가능, 저자 표기 조건). 단, **IT 엔드포인트/네트워크 로그 편향** — 드론 RF/항법 텔레메트리 시그니처는 거의 없음. → 관측→ATT&CK 브리지로 선택적 활용, sim 텔레메트리와는 도메인 갭.

**드론/UAV 학술 위협 분류** — 예: Frontiers(2026) UAV 딥러닝 보안 리뷰는 **OSI 7계층 taxonomy**로 공격을 정리하고, **각 공격에 대응하는 관측 anomaly를 직접 표로 매핑**:
- GPS 스푸핑 → Loss of GPS Lock, Altitude Drift, Heading Instability
- 재밍(Jamming) → Packet Drop Burst, Link Latency Spikes, Unexpected Retransmissions
- DoS → Routing Loop, Unreachable Host, Network Congestion
- Session hijacking → Port Unavailability, Connection Timeout, Retransmission Burst

이는 **정확히 RAG-A가 필요로 하는 "sim 관측치 → 공격유형" 매핑**이며 ATT&CK 본문엔 없는 정보. (단 대부분 ATT&CK ID를 쓰지 않으므로, ATT&CK/ICS/Mobile ID로의 크로스워크는 수작업 정형화 필요.) Mississippi State 석사논문 등 "MITRE ATT&CK의 UAV 적용" 학술 연구는 UAV에 유의미한 ATT&CK 기법을 선별·매핑하는 작업을 수행(원문 직접열람 403, 초록·2차 인용 기반 — 추정). → **관측→공격 매핑의 핵심 보완 소스.**

**CTI 피드** — 실제 위협 최신성(novel의 실사례) 보완엔 유리하나, 신뢰도·형식 편차와 운영 부담 큼. 해커톤 범위에선 우선순위 낮음(선택).

### 1.3 추천 데이터 조합 (RAG-A)

**계층형 결합 — 3층 구조 권장:**

1. **백본(권위·서술)**: **ATT&CK Enterprise + ICS + Mobile** 3개 매트릭스 통합 인덱싱.
   - Enterprise = IT 침투/C2/유출/펌웨어, ICS = cyber-physical 영향·제어상실, Mobile = RF/GPS/radio DoS. 세 도메인 합치면 드론 공격 표면을 실질적으로 커버. 모두 STIX/JSON·동일 스키마·무료 상용가능·D3FEND 연동.
2. **일반화 층(novel 대응)**: **CAPEC** 결합. 관측이 기존 기법 어느 것과도 안 맞을 때 상위 공격패턴으로 일반화. CWE↔ATT&CK 크로스워크로 취약점 근거 보강.
3. **관측→공격 매핑 층(RAG-A 판별의 핵심)**: **드론/UAV 학술 taxonomy(OSI 계층 anomaly↔attack 표)** 를 정형화해 인덱싱. 여기에 각 항목을 ATT&CK/ICS/Mobile ID로 매핑한 **소규모 큐레이션 크로스워크**를 추가. 선택적으로 **Sigma 태그**를 IT 로그 관측의 보조 브리지로.

> 요지: **ATT&CK 3-도메인 = "무슨 공격인지"의 어휘/서술**, **드론 taxonomy = "이 징후가 어느 공격인지"의 매핑**. 둘을 함께 넣어야 "SNR↓·GPS 이상 → 기법 ID" 질의가 실제로 검색됨. Enterprise 단독은 어휘의 절반만 제공.

---

## 2. 리서치 ② — 최적 임베딩 모델

### 2.1 모델 비교표

평가축: MTEB retrieval / 파라미터·컨텍스트 / 한국어 지원 / 로컬 실행성 / 라이선스(상용) / 도메인 적합성.

| 모델 | 파라미터 | 컨텍스트 | MTEB retrieval | 한국어 | 로컬 실행 | 라이선스 | 비고 |
|---|---|---|---|---|---|---|---|
| **BGE-M3** (BAAI) | ~568M | **8192** | 다국어 SOTA급 | **양호**(100+ 언어) | GPU 권장, CPU 가능 | **MIT** (상용OK) | dense+**sparse**+multi-vec 하이브리드 |
| **KURE-v1** (고려대 nlpai-lab) | ~568M (bge-m3 기반) | 8192 | 한국어 retrieval **1위급**(MTEB-ko) | **최상** | bge-m3와 동일 | MIT (추정, KoE5=MIT) | bge-m3 한국어 파인튜닝, drop-in |
| **multilingual-e5-large** | 335M | 512 | 강함(다국어) | 양호 | 경량·CPU 수월 | **MIT** | 512 토큰 한계 |
| **gte-multilingual-base** (Alibaba) | 305M | 8192 | 다국어 강함 | 양호 | 경량 | Apache-2.0 | 크기/성능 균형 |
| **KoE5** (nlpai-lab) | 560M | 512 | 한국어 강함 | 최상 | e5 기반 | **MIT** | ml-e5 한국어 파인튜닝 |
| bge-large-en-v1.5 | 335M | 512 | 영어 강함 | ✗ | 경량 | MIT | 영어전용 |
| e5-large-v2 | 335M | 512 | 영어 강함 | ✗ | 경량 | MIT | 영어전용 |
| all-mpnet-base-v2 (SBERT) | 110M | 384 | 중간 | ✗ | CPU 매우 수월 | Apache-2.0 | 구세대 베이스라인 |
| all-MiniLM-L6-v2 (SBERT) | 22M | 256 | 낮음 | ✗ | CPU 초경량 | Apache-2.0 | 프로토타입/속도용 |
| Qwen3-Embedding (0.6B/4B/8B) | 0.6~8B | 8K+ | **최상위권**(다국어) | 최상 | 0.6B는 로컬 가능, 8B 무거움 | Apache-2.0 | 최신 고성능 대안 |
| NV-Embed-v2 (NVIDIA) | 7.85B | 32K | MTEB 최상위 | 양호 | **무거움**(대용량 GPU) | **CC-BY-NC**(비상용) | 라이선스·크기 제약 |
| voyage-3-large / OpenAI-3-large | API | - | 최상위 | 양호 | **로컬 불가(API)** | 상용 API | 내부데이터 업로드 금지 정책과 충돌 |

### 2.2 추천 및 근거

**1순위: BGE-M3**
- **다국어(영문 ATT&CK 설명 + 한국어 쿼리) 단일 모델로 동시 처리** — 영어 코퍼스/한국어 질의 크로스링구얼 검색에 적합.
- **하이브리드(dense + sparse/lexical + multi-vector)**: RAG-A는 `T1464`, `GPS`, `SNR`, `jamming`, `deauth` 같은 **정확 키워드·기법 ID 일치**가 중요한데, sparse(lexical) 성분이 이 exact match를 dense 의미검색과 함께 잡아줌 → 기법 ID 검색 정확도에 직접 기여.
- **8192 토큰 컨텍스트**: ATT&CK 기법 전체 서술·절차 예시를 한 청크로 임베딩 가능.
- **MIT 라이선스**(상용·방산 활용 자유), 로컬 실행(단일 GPU 무난, CPU도 가능). MTEB 다국어 retrieval SOTA급.

**한국어 쿼리 비중이 높다면: KURE-v1 (고려대 nlpai-lab)**
- bge-m3를 한국어 검색에 파인튜닝 → 한국어 retrieval MTEB-ko 리더보드 최상위, 대부분 다국어 모델 압도. **아키텍처가 bge-m3와 동일**하므로 BGE-M3와 **무손실 drop-in 교체** 가능(파이프라인 재작성 불필요). 라이선스 MIT(추정; 동 연구실 KoE5는 MIT 확인). → 한/영 혼합 쿼리라면 KURE-v1과 BGE-M3를 A/B 비교 후 채택 권장.

**경량/CPU 여건: multilingual-e5-large 또는 gte-multilingual-base**
- 335M/305M로 가볍고 다국어·한국어 지원, MIT/Apache-2.0. 단 e5는 512 토큰 한계(긴 ATT&CK 서술은 청킹 필요), gte는 8192 지원해 gte가 유리.

**성능 상한을 원하면(자원 여유 시): Qwen3-Embedding-0.6B/4B**
- 최신 다국어 최상위권, Apache-2.0(상용OK). 0.6B는 로컬 가능. BGE-M3 대비 품질 이득이 필요할 때 후보.

**회피 권장**
- **NV-Embed-v2**: MTEB 최상위지만 7.85B로 무겁고 **CC-BY-NC(비상용)** → 방산/상용 부적합.
- **voyage/OpenAI/Cohere API**: 성능 우수하나 로컬 불가·API 호출 = "내부데이터 업로드 금지" 제약과 충돌.
- **all-MiniLM-L6-v2**: 초경량 프로토타입엔 좋으나 한국어 불가·품질 낮아 최종 RAG-A엔 부적합.

> **최종 추천**: 기본 **BGE-M3**(하이브리드 검색 활성화: dense+sparse), 한국어 질의 중심 운영 시 **KURE-v1**로 교체. 둘 다 MIT·bge-m3 계열이라 리스크 없이 상호 전환. 자원 제약 시 **gte-multilingual-base**로 다운스케일.

---

## 3. 핵심 발견 (재정리)

1. **Enterprise 단독 불충분 → 3-도메인(Enterprise+ICS+Mobile) 통합이 최소 요건.** 재밍/GPS는 Mobile(T1464), cyber-physical 영향은 ICS에 존재. 세 도메인은 동일 STIX 스키마·무료·D3FEND 연동이라 통합 비용 낮음.
2. **RAG-A의 진짜 병목은 "관측→공격" 매핑 데이터.** ATT&CK 본문은 "공격이 무엇인지"는 잘 서술하나 "SNR↓·GPS lock 상실 → 어느 기법"은 담지 않음. 이 매핑은 **드론 학술 taxonomy(OSI anomaly↔attack 표)** 와 **Sigma 태그**가 제공 → 반드시 보완.
3. **Novel 대응**엔 CAPEC의 상위 공격패턴 추상화 + BGE-M3의 의미 유사도 검색 조합이 유효(정확 기법 없을 때 상위 패턴/유사 서술로 폴백).
4. **임베딩은 BGE-M3 계열이 정답에 가깝다.** 다국어(EN 코퍼스+KO 쿼리)·MIT·8K·하이브리드(기법 ID exact match)라는 RAG-A 요구가 정확히 맞물림. 한국어 특화가 필요하면 동일 계열 KURE-v1.
5. **하이브리드 검색(dense+sparse) 활성화 권장** — 기법 ID/전문 용어 exact match가 판별 정확도에 크게 기여.

---

## 4. 출처 (공개 자료)

**데이터 소스**
- MITRE ATT&CK ICS Matrix — https://attack.mitre.org/matrices/ics/
- MITRE ATT&CK Mobile — Network DoS(T1464, GPS/radio 재밍) — https://attack.mitre.org/techniques/T1464/ , https://attack.mitre.org/matrices/mobile/
- MITRE ATT&CK for ICS 설명(IT→OT, 물리적 영향) — https://www.denexus.io/learn/articles/mitre-attck-for-ics-explained-tactics-techniques-and-cross-domain-attack-paths
- ATT&CK Terms of Use(무료·상용 라이선스) — https://attack.mitre.org/resources/legal-and-branding/terms-of-use/
- CAPEC List v3.9 — https://capec.mitre.org/data/index.html ; CWE↔CAPEC↔ATT&CK 매핑 — https://www.nopsec.com/blog/mapping-cves-and-attck-framework-ttps-an-empirical-approach/
- Sigma(SigmaHQ) 저장소·DRL 1.1 — https://github.com/SigmaHQ/sigma , https://github.com/SigmaHQ/Detection-Rule-License
- Sigma↔ATT&CK 태깅 자동화(관측→공격 브리지) — https://socprime.com/blog/uncoder-ai-automates-mitre-attck-tagging-in-sigma-rules/
- UAV 위협 taxonomy(OSI 계층, 관측 anomaly↔attack 매핑; MITRE 미사용) — Frontiers in AI 2026 — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1752124/full
- UAV cyber 위협/GPS 스푸핑·재밍 분류 리뷰 — https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ise2/2046868 , https://www.mdpi.com/2504-446X/9/10/682
- MITRE ATT&CK의 UAV 적용(석사논문, 원문 403·2차인용 기반, 추정) — https://scholarsjunction.msstate.edu/ (article=7217)

**임베딩 모델**
- BGE-M3(568M·8192·dense+sparse+multivec·MIT) — https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models , https://medium.com/@mrAryanKumar/comparative-analysis-of-qwen-3-and-bge-m3-embedding-models-for-multilingual-information-retrieval-72c0e6895413
- KURE-v1(bge-m3 한국어 파인튜닝, 고려대) — https://huggingface.co/nlpai-lab/KURE-v1 , https://github.com/nlpai-lab/KURE
- KoE5(MIT, ml-e5 한국어) / MTEB-ko-retrieval 리더보드 — https://huggingface.co/nlpai-lab/KoE5
- Multilingual E5(335M·512) 기술 리포트 — https://arxiv.org/pdf/2402.05672
- gte-multilingual-base(305M·8192) — https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
- MTEB 리더보드/모델 비교(2025~26, NV-Embed·BGE·E5·Qwen3) — https://futureagi.com/blog/best-embedding-models-2025/ , https://modal.com/blog/mteb-leaderboard-article , https://app.ailog.fr/en/blog/guides/choosing-embedding-models

> 표기: MTEB 세부 점수·일부 파라미터/라이선스(KURE-v1 MIT 등)는 2차 출처 기반 "추정" 포함. 확정 필요 시 HuggingFace 모델 카드와 MTEB 공식 리더보드 원본 재확인 권장.
