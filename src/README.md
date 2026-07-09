# NeuroGuard — 드론군집 사이버 방어 (통합 README)

> 방어 모델 실행법 + 모델 차이 + 파일 지도. **정리(무엇을 지울지) 판단용.**

---

## TL;DR (3줄)
- **메인 방어 모델 = `agents/jy_hvt.py` (HVT)** — belief-검증-트리거, 완전 구현, 자체 작동.
- **RAG(공격탐지+방어채택)는 별도 실험** — sim과 다른 env, 점수 경로와 **오프라인으로만 연결**(appendix용).
- **채점 러너는 junyeong(개인)에** 있음. 팀 src엔 모델 코드 + rule/llm/rl 러너만 있음.

---

## 1. 환경 2개 (왜 분리)
| 환경 | 경로 | 용도 | 핵심 |
|---|---|---|---|
| **sim** | `C:\workspace\dah_venv` | 방어 모델(hvt·reach2·rag_guided) 채점 | CybORG + **numpy 1.23 고정** |
| **RAG** | sentence-transformers 별도 venv | RAG-A·RAG-B 실행 | 최신 numpy/torch 필요 |

⚠️ **둘은 한 프로세스에서 못 섞임** → RAG와 sim은 항상 따로 실행.

---

## 2. Quick Start

> 방어 모델 실행엔 **CybORG env**(numpy 1.23) 필요 — 예: `C:\workspace\dah_venv`.

### ★ ① 방어 모델 채점 — `src/score.py` (팀 자립 실행, 권장)  — CybORG env
```bash
cd C:/workspace/DAH-2026/src
python score.py --model hvt --scenario A17 --recall 0.75 --fp 0.1 --seeds 5
#  → 방어 점수 = 0.900
```
- `--model`: **hvt**(메인) · reach2 · rag-guided
- `--scenario`: A1~A21 · A-CONN · A-MV
- 실전조건: `--recall 0.75 --fp 0.1` (미탐/오탐). 생략 시 오라클(1.0/0.0).
- **step별 상태 로그:**
  ```bash
  python score.py --model hvt --scenario A17 --log steps.csv
  #  → steps.csv: step, compromise_frac(감염률), availability(가용성), n_compromised
  ```
> junyeong 없이 **팀 src만으로** 우리 모델 채점 + step별 상태까지 됨. (junyeong 하네스와 점수 동일 확인)

### ② RAG 파이프라인 데모(appendix)  — RAG env (별도!)
```bash
pip install -r attack_rag/requirements.txt      # 최초 1회, sentence-transformers 별도 env
cd C:/workspace/DAH-2026/src
python -m attack_rag.integration_test
#  → 관측 → 공격유형(RAG-A) → 방어권고(RAG-B) 출력 (여기서 끝, sim 실행 아님)
```

### ③ (참고) 팀 기존 러너 run.py — rule/llm/rl 전용
```bash
python run.py --red rule --blue rule --scenario A1     # log.csv에 step별 상태 기록
```
> `--blue`는 {rule,llm,rl}만. 우리 모델(hvt)은 여기 안 물림 → ①번 `score.py` 사용.

---

## 3. 모델 차이 (헷갈리는 핵심)

### 방어 모델 (sim env에서 실행)
| 파일 | 클래스 | 무엇 | 점수 | 비고 |
|---|---|---|:--:|---|
| **`agents/jy_hvt.py`** | HVTDefense | **HVT — belief 예측 + 반사실 검증(Δ>τ)만 재장악** | **~0.92** | ★**메인**. 완전 구현·자체작동 |
| `agents/reach2.py` | ReachV2 | 탐지된 감염 **전부** 재장악 + de-jam + relay | ~0.90 | 단순·견고. rag_guided의 base |
| `agents/rag_guided.py` | RAGGuidedPolicy | **reach2 + attack_class 자세라우팅**(재밍=FP회피) | ~0.93 | RAG 실험. **reach2 상속**(HVT 아님) |
| `agents/defense_base.py` | DefensePolicy + 헬퍼 | 공용 인터페이스·adjacency·retake 등 | — | 위 3개가 공유하는 부품 |

**핵심 차이 = "재장악을 얼마나 가려서 하느냐":**
- reach2 = 탐지된 것 다 재장악 (오탐도) → 낭비.
- HVT = 반사실 검증 통과분만 재장악 → 오탐 회피(더 정교).
- rag_guided = reach2 + "재밍이면 오탐 재장악 끄기" → HVT의 오탐회피를 재밍에 한정해 흉내.

> ⚠️ **이름 주의:** "HVT+RAG"라 불렀지만 rag_guided의 코드 base는 **reach2**(jy_hvt 아님). 순수 HVT는 jy_hvt.py.

### RAG (RAG env에서 실행)
| 파일 | 클래스 | 무엇 |
|---|---|---|
| `attack_rag/rag_a.py` | RagA | **RAG-A** — 관측 → 공격유형(ATT&CK) + attack_class |
| `defense_rag/pipeline.py` | DefenseRAG | **RAG-B**(박수민) — 공격유형 → 방어권고(D3FEND/NIST) |
| `attack_rag/integration_test.py` | — | RAG-A→RAG-B **데모** (권고 출력하고 끝) |

---

## 4. ⚠️ 정직한 연결 상태 (안 이어진 부분)
```
[RAG 경로]  관측 → rag_a → defense_rag → 방어"권고"     (RAG env, 여기서 끝)
                                            ✂ 자동연결 없음
[sim 경로]  rag_guided → reach2 → 방어실행 → 점수        (sim env)
                 ▲ attack_class를 "미리 계산된 파일"(scenario_attack_class.json)에서 읽음
```
- **integration_test는 rag_guided를 안 부름.** 두 경로는 별개 실행.
- **rag_guided는 defense_rag(RAG-B)를 안 씀** — RAG-A의 attack_class만 오프라인 파일로 소비.
- 즉 "0.93"은 reach2 + 미리계산 class + 하드코딩 재밍로직. **RAG가 방어를 실시간 구동하는 게 아님.**
- 원인: RAG env ↔ sim env 프로세스 분리 → 라이브 연결 불가. 라이브 파이프라인은 **본선/후속 과제.**

**→ 그래서 메인은 jy_hvt(독립완결), RAG는 appendix가 맞음.**

---

## 5. 파일 지도 — 정리 판단용

### GitHub main (팀 공유)
```
src/
├─ agents/
│   ├─ jy_hvt.py         ★ 유지 — 메인 모델
│   ├─ reach2.py         유지 — rag_guided가 의존
│   ├─ rag_guided.py     RAG 실험 (appendix면 유지, 아니면 reach2와 함께 정리 고려)
│   ├─ defense_base.py   유지 — 공용 부품
│   └─ (rule/hier/sm_/react … 팀 기존 모델들)   ← 팀 판단
├─ attack_rag/           RAG-A (appendix) — rag_a·data·docs·차트
├─ defense_rag/          RAG-B (박수민, appendix)
└─ run.py · sweep.py · sim/ · scenarios/ · configs/   유지 — 팀 sim 엔진
```

### 정리 시 체크포인트
| 항목 | 판단 |
|---|---|
| jy_hvt·reach2·defense_base | **유지** (메인 + 의존) |
| rag_guided | appendix로 남길지 / 점수경로 아니니 뺄지 — **팀 결정** |
| attack_rag·defense_rag | appendix면 유지. 실측 재현 원하면 유지 |
| docs 차트(model_comparison·scenario_dynamics 등) | appendix 근거로 유지 권장 |
| junyeong `neuroguard/` (하네스·전체모델·1v1) | **본선 private** — 팀 레포엔 안 올림 |

### 채점 러너
- **`src/score.py`** ✅ — 우리 모델(DefensePolicy)용 rollout+점수 러너. **팀 src만으로 `python score.py --model hvt` + step별 로그 가능** (junyeong 불필요). 시나리오 spec = `src/configs/attack_scenarios.yaml`.

---

## 6. 한 장 요약
| 질문 | 답 |
|---|---|
| 메인 모델? | `agents/jy_hvt.py` (HVT, 자체완결) |
| rag_guided가 HVT? | ❌ reach2 상속 ("HVT+RAG"는 느슨한 이름) |
| RAG가 방어 실시간 구동? | ❌ 오프라인 파일로만 연결 (appendix) |
| 팀이 src로 우리 모델 채점? | ✅ `python score.py --model hvt` (자립) |
| step별 상태 볼 수 있나? | ✅ run.py는 자동 log.csv, 우리 모델은 뽑으면 됨 |
