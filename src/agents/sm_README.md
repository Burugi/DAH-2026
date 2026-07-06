# sm_*.py — 방어 모델 구조 모음 (팀원 구조 파악용)

이 세션에서 실험한 방어(blue) 아키텍처들의 **구조 사본**입니다. 각 `sm_{모델명}.py`는 해당
모델의 클래스 + 헬퍼 + 설명 헤더를 담아 **단독으로 읽히도록** 구성했습니다.
※ 읽기 전용 — 실제 실행/임포트는 원본(`experimental.py`/`multiagent.py`/`brains.py`)이 담당.

## 공통 인터페이스 (모든 모델이 `BlueBrainBase` 상속)
- `team_decide(ctx, agents)` → **드론별 행동 id 리스트** (핵심. 18대에 각각 배정)
- `step_decide(ctx)` → 팀 단일 행동 (per-agent 경로용)
- `recovery_priority(ctx, agents, aids)` → 예산 초과 시 어떤 회복행동을 남길지 우선순위

> 구조 = **중앙 브레인 1개**가 전체 상황(ctx)을 보고 18대 드론 행동을 한 번에 결정하는 **중앙집중식**.
> (드론마다 독립 에이전트를 두는 분산식이 아님.)

## 참조표 1 — 행동 id (0~9)
| id | 이름 | 뜻 |
|---|---|---|
| 0 | Sleep | 아무것도 안 함 |
| 1 | Monitor | 감시(상태변화 없음) |
| 2 | Analyse | 자기 드론 세션 분석 |
| 3 | RemoveSessions | **자가치유** — 자기 드론의 red 세션 제거(약한 원시액션) |
| 4 | RetakeSuspicious | **탈환** — 감염 드론 재이미징(강한 액션) |
| 5 | RetakeRandom | 임의 드론 탈환 |
| 6 | BlockSuspicious | 감염 드론 트래픽 차단 |
| 7 | AllowTraffic | 차단 해제 |
| 8 | DeployDecoy | 디코이(오프라인 빌드선 관측으로 처리) |
| 9 | Failsafe | 로컬 자율방어(감염 시 자가치유, 아니면 링크 복구) |

**예산 캡:** 스텝당 능동봉쇄 `{3,4,5,6}` 수를 k로 상한. 초과분은 `recovery_priority` 낮은 순 Monitor 강등.

## 참조표 2 — ctx 키 (관측 입력)
| 키 | 뜻 |
|---|---|
| compromised | 감염된 드론 id 집합 |
| n | 드론 수(18) |
| pos | 드론별 위치 `[x, y]` |
| snr | 신호세기 (낮을수록 재밍당함) |
| gps_err | GPS 오차 (높을수록 스푸핑) |
| link_up | 드론별 연결 여부(bool) |
| max_link | 연결 판정 반경 |
| ip_to_drone | IP→드론 id 매핑 |

## 22개 모델 한 줄 요약
### 베이스라인
- **sm_rule** — 자기 감염→치유(3)/감염 보이면→탈환(4)/없으면→감시(1). 클래스 아닌 per-drone FSM.

### O-P-E-R 루프 (관찰-계획-실행-반성)
- **sm_react** — 관찰→추론(CoT)→행동, 위협등급별 대응
- **sm_reflect** — 행동 K회마다 전략 태그 반성·갱신
- **sm_ooda** — 관찰-판단-결정-행동, 위협수준×공격모드 정책테이블
- **sm_plan** — 에피소드 시작에 위상 계획, 임계 초과 시 재계획

### 창의형
- **sm_predictive** — SIR 확산예측, 감염 front 인접만 탈환 + 원거리 보존
- **sm_graph** — 그래프 중심성 휴리스틱(GNN-lite), 허브/중간/leaf 역할 배분
- **sm_rag** — 5D 시그니처→하드코딩 플레이북 카드 최근접 검색(RAG-lite)
- **sm_bandit** — 컨텍스추얼 밴딧, 스탠스 가치를 보상으로 학습(ε-greedy)
- **sm_ensemble** — 전문가 혼합(MoE) graph·predictive·rule 드론별 다수결
- **sm_hybrid** — rule 우선순위 + rl 보상EMA + llm 위협tier 융합
- **sm_debate** — Hawk(봉쇄) vs Dove(보존) 페르소나 토론→심판 채택

### 할당형 (누구를 회복하나)
- **sm_auction** — 시장기반 경매(CBBA), 건강 드론이 감염 태스크 입찰
- **sm_whittle** — 휘틀/cμ 인덱스(전파수/탈환비용) 큰 순 배정
- **sm_mincut** — 그래프 최소절단, 감염-건강 경계 우선 절단

### 분산·선제형
- **sm_riskfield** — 베이지안 위험장, 고위험 미감염에 선제 디코이/차단
- **sm_stigmergy** — 페로몬 스티그머지(개미군집), 로컬 농도 기반 분업
- **sm_msgpass** — 로컬 GNN 메시지패싱, 이웃 피처 k라운드 집계(로컬 관측만)

### 자동최적화 + 이 세션 핵심 변형
- **sm_evo** — 진화전략으로 임계값 θ(8D) 자동최적화한 반사정책 (무제약 상한 참조)
- **sm_evo_k6** — 위 EvoBlue의 k=6 최적화 θ 판 (예산=6 상한 참조)
- **sm_whittle_sh** — whittle + **자가치유 우선** 역이식 (= WhittleBlue selfheal=1.0)
- **sm_graph_sh** — graph + **자가치유 우선** 역이식 = WhittleBlue와 동일 원리. **최종 권장 아키텍처**
  (red 저상승 조건). GraphCentralityBlue selfheal=1.0.

> `_sh`/`_k6`는 **별도 클래스가 아니라 base 클래스의 파라미터판**입니다 — 구조는 base 파일과 동일,
> 차이는 헤더에 명시. 자가치유 항은 예산이 걸릴 때만 발동(무제약 k=∞에선 원판과 동일).

## 결과 서사 (한 줄)
무제약이면 아키텍처가 안 갈림(단일축) → 예산 제약이 할당 지능 노출(3구간) → evo로 상한 추정 →
evo가 찾은 "자가치유 우선"을 graph/whittle에 역이식(n=2) → graph_sh가 evo 상한과 동점.
단 red 완전 상승 시 자가치유 우위 소멸(유효 경계). 자세히는 `docs/final_report.md`.
