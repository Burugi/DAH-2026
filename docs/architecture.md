# Architecture

CybORG DroneSwarm(CAGE Challenge 3) 위의 실험 하네스. 기본 실행 모델은 **HVT+RAG(rag-guided)** —
RAG-A의 attack_class로 대응 자세를 라우팅하고 **HVT** 계열 방어정책이 실행한다.
`rule`/`llm`/`rl` 3×3 비교를 베이스라인으로 함께 제공한다.

## 전체 파이프라인

공격 관측(raw) → RAG-A 공격타입 판단(ATT&CK KB) → RAG-B 방어책 검색(D3FEND KB)
→ HVT 에이전트 → CybORG 실행.

![RAG 파이프라인](diagrams/rag_flowchart.png)

HVT 방어 내부 — 관측층(텔레메트리·탐지기) → 두뇌(belief + world-model 반사실 검증) → 실행층.

![HVT 구조](diagrams/hvt_architecture.png)

(HVT 단계별 플로우차트: `diagrams/hvt_flowchart.png`)

## 두 채널 설계

CybORG는 드론 네트워크(세션·익스플로잇·통신 링크)를 모델링하지만 **GPS/RF 물리는 없다**.
그래서 하나의 함대 위에 두 채널을 정렬해 돌린다.

| 채널 | 생성 | 신호 |
|---|---|---|
| 네트워크 점령 (exploit / seize / worm) | CybORG 시뮬 | `red_owned`, `reward` |
| 위치 / GPS / 통신 (재밍·스푸핑) | 합성 레이어 `sim/fleet.py` | `snr`, `link_up`, `gps_err`, 공격 라벨 |

합성 함대의 초기 배치가 시뮬 시작 위치(`starting_positions`)를 시딩하므로 두 채널은 같은
함대를 기술한다. **에이전트 비교는 네트워크 채널에서** 이뤄지고, 재밍/GPS 스푸핑은 합성 시나리오
컨텍스트이며 패시브 텔레메트리 방어는 정답 라벨로 채점된다(`sim/defense.py`).

## HVT 방어

`src/agents/hvt.py` — Hypothesis-Verify-Trigger. belief 상위를 무조건 재장악하지 않고, 각 후보의
재장악 실익을 world-model 반사실 시뮬로 검증해 트리거를 넘는 것만 파괴적으로 재장악한다.

1. **가설** — belief b[i]=P(감염)를 웜확산 전진예측 + 탐지 융합으로 유지(관측 detected + 인접만 사용).
2. **검증** — 후보 i에 대해 Δ_i = "지금 i를 재장악하면 앞으로 H스텝 막는 확산량"을 반사실로 계산.
3. **트리거** — Δ_i > τ 인 타겟만 파괴적 재장악(RetakeControl), 나머지는 비파괴 자가치유로 hold.
4. **실행** — 재장악을 Δ 큰 순으로 다중배정, relay·de-jam 유지.

`src/agents/defense_base.py`가 `DefensePolicy` 베이스와 그래프 유틸(adjacency/components/retake_target)을
제공하고, `src/score.py`가 시나리오 spec으로 환경을 세워 방어 모델을 구동·채점한다.

## 방어 모델 비교

DefensePolicy 인터페이스(reset/step)를 따르는 방어 모델들. 핵심 차이는 "재장악을 얼마나 가려서 하느냐"다.

| 파일 | 클래스 | 방식 | 방어점수 |
|---|---|---|:--:|
| `agents/hvt.py` | HVTDefense | belief 예측 + 반사실 검증(Δ>τ) 통과분만 재장악 → 오탐 회피 | ~0.92 (메인) |
| `agents/reach2.py` | ReachV2 | 탐지된 감염 전부 재장악 + de-jam + relay (단순·견고) | ~0.90 |
| `appendix/rag_guided.py` | RAGGuidedPolicy | reach2 + 공격유형 자세 라우팅(재밍이면 오탐 재장악 끔) | ~0.93 |

`rag_guided`는 reach2를 상속하며(HVT 아님) 공격유형(attack_class)을 오프라인 파일에서 읽는다. RAG
파이프라인(RAG-A·RAG-B)은 sentence-transformers 계열의 별도 env에서 방어 "권고"까지만 생성하고 sim을
실시간 구동하지 않으므로, RAG 관련 코드는 모두 `appendix/`에 둔다.

## 파이프라인

```
방어 모델(DefensePolicy)                        rule/llm/rl 베이스라인
        │                                              │
configs/attack_scenarios.yaml                    configs/sweep.yaml
        │                                              │
        ▼                                              ▼
 score.py / record.py ──spec──► defense_base.DefensePolicy   sweep.py ── 9 매치업 (red×blue)
        │              (RAGGuidedPolicy·HVTDefense·ReachV2)     │
        └────────► run.build_env() ─► CybORG DroneSwarm + 합성 함대(fleet.py)
                                                        │
        record.py/sweep.py ─► run.save_run() ─► results/<run>/ {log.csv, arrays.npz, meta.json}
                                                        │
                                     ┌──────────────────┼──────────────┐
                                     ▼                  ▼              ▼
                                dashboard.py        render.py       plot.py
                                (HTML+PNG)        (animation.gif)  (static figs)
```

`score.py`는 방어점수(+옵션 step CSV)만 출력하고, `record.py`가 같은 rollout을 `results/<run>/`
번들로 저장해 대시보드/애니메이션 도구를 그대로 쓸 수 있다(`docs/demo.md` 참고). `docs/sample_run/`은
그렇게 만든 HVT A1 예시다.

## 핵심 모듈

- **`agents/actions.py`** — red/blue 행동 카탈로그 단일 소스. 각 행동을 실제 DroneSwarm 프리미티브
  또는 합성 방어에 매핑하고 MITRE 태그를 붙인다. LLM 텍스트 메뉴 포함.
- **`agents/brains.py`** — rule/llm/rl 세 에이전트 타입(양 진영 공통 카탈로그).
- **`agents/hvt.py` · `agents/reach2.py` · `agents/defense_base.py`** — HVT/단순 재장악 방어와 공용 베이스.
- **`agents/hierarchical.py` · `agents/multiagent.py`** — 계층형/멀티에이전트 blue 브레인(react/reflect/plan/ooda).
- **`agents/llm.py`** — 기본은 오프라인 stub, `ANTHROPIC_API_KEY`가 있으면 Claude.
- **`agents/rl.py`** — 순수 numpy tabular Q(1회 학습 후 캐시).
- **`sim/fleet.py` · `sim/defense.py`** — 합성 UAV+UGV 텔레메트리 / 채점되는 텔레메트리 방어.
- **`score.py`** — DefensePolicy 방어 모델 채점 러너 (기본 모델 rag-guided).
- **`record.py`** — 방어 모델 실행을 `results/<run>/` 대시보드 번들로 기록 (데모용).
- **`viz/dashboard.py`** — 단일 HTML 대시보드(맵·전술 로그·점수 곡선) + 프리뷰 PNG.
- **`viz/plot.py` · `viz/render.py` · `viz/score.py`** — 정적 그림 / 애니메이션 / 종합점수.

## Blue 행동 id (0~9)

방어 브레인이 반환하는 행동 id. `agents/actions.make_blue_index()`가 CybORG 래퍼 인덱스로 매핑한다.

| id | 이름 | 뜻 |
|---|---|---|
| 0 | Sleep | 아무것도 안 함 |
| 1 | Monitor | 감시(상태 변화 없음) |
| 2 | Analyse | 자기 드론 세션 분석 |
| 3 | RemoveSessions | 자가치유 — 자기 드론의 red 세션 제거(비파괴) |
| 4 | RetakeSuspicious | 탈환 — 감염 드론 재이미징(파괴적) |
| 5 | RetakeRandom | 임의 드론 탈환 |
| 6 | BlockSuspicious | 감염 드론 트래픽 차단 |
| 7 | AllowTraffic | 차단 해제 |
| 8 | DeployDecoy | 디코이 |
| 9 | Failsafe | 로컬 자율방어(감염 시 자가치유, 아니면 링크 복구) |

## ctx 키 (관측 입력)

| 키 | 뜻 |
|---|---|
| compromised | 감염된 드론 id 집합 |
| n | 드론 수 |
| pos | 드론별 위치 `[x, y]` |
| snr | 신호세기(낮을수록 재밍) |
| gps_err | GPS 오차(높을수록 스푸핑) |
| link_up | 드론별 연결 여부 |
| max_link | 연결 판정 반경 |
| ip_to_drone | IP → 드론 id 매핑 |

## counterpart_models / appendix

`src/counterpart_models/`는 실험 과정에서 만든 대안 방어 모델들의 **참조 사본**이다(비실행). HVT와
같은 `DefensePolicy`/`BlueBrainBase` 인터페이스를 따르는 belief·frontier·graph·auction·bandit 등
계열의 구조를 담는다.

`appendix/`는 RAG 파이프라인을 보관한다 — `attack_rag/`(RAG-A, 공격유형 판단), `defense_rag/`(RAG-B,
방어권고), `rag_guided.py`(HVT+RAG 통합 — `score.py`/`record.py`의 **기본 모델**로 실행되며 RAG-A의
오프라인 산출물만 읽으므로 임베딩 env가 필요 없다), `rag_playbook.py`(시그니처 최근접 플레이북 검색
RAG-lite). 임베딩·KB 검색(RAG-A/B 파이프라인 자체)은 sentence-transformers 계열의 별도 env에서 돈다.
