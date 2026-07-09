# DroneSwarm 공·방 Lab

CybORG DroneSwarm(CAGE Challenge 3) 위에서 드론 군집의 사이버 공격(red)·방어(blue)를 시뮬레이션하는
실험 도구임. 중심 방어 모델은 HVT(Hypothesis-Verify-Trigger)이고, `rule`·`llm`·`rl` 3×3 비교를
베이스라인으로 제공함.

- 환경: UAV 12대 + UGV 6대, 100×100 격자.
- 공격 시나리오 A1~A21은 `src/configs/attack_scenarios.yaml`에 정의(MITRE ATT&CK/D3FEND 태그).

## 실험 설계

- **두 채널** — CybORG가 네트워크 점령(익스플로잇·장악·웜 확산)과 보상을 담당함. CybORG에는 GPS/RF
  물리가 없으므로 위치·통신·재밍·GPS 스푸핑은 합성 텔레메트리 레이어(`sim/fleet.py`)가 정답 라벨과
  함께 생성함. 합성 함대의 초기 배치가 시뮬 시작 위치를 시딩하므로 두 채널은 같은 함대를 기술함.
- **행동** — 매 스텝 red/blue가 MITRE 태그가 붙은 고정 카탈로그에서 선택함. red는 정찰·익스플로잇·
  장악·확산·재밍·차단, blue는 감시·분석·세션제거·탈환·차단·자율방어로 구성됨(`src/agents/actions.py`).
- **평가** — 시나리오 × 시드 평균으로, 진영별 단일 종합점수(0~1)를 산출함. 공격 A = 평균(점령률·지속도·
  침투속도·은밀성), 방어 D = 평균(봉쇄·가용성). 탐지 F1(comp/jam/gps)과 가용성도 함께 기록함.

## HVT 방어

기존 belief·frontier 방어는 탐지 상위를 무조건 파괴적으로 재장악해, 현실 탐지(오탐·미탐)에서 청정
노드까지 재장악하며 대응 자원을 낭비함. HVT는 각 후보의 재장악 실익을 world-model 반사실 시뮬로
검증해 트리거를 넘는 것만 파괴적으로 재장악함.

1. **가설** — belief b[i]=P(감염)를 웜확산 전진예측 + 탐지 융합으로 유지함(관측 detected와 인접만 사용).
2. **검증** — 후보 i에 대해 Δ_i = "지금 i를 재장악하면 앞으로 H스텝 막는 확산량"을 반사실로 계산함.
3. **트리거** — Δ_i > τ 인 타겟만 파괴적 재장악(RetakeControl)하고, 나머지는 비파괴 자가치유로 hold함.
4. **실행** — 재장악을 Δ 큰 순으로 다중배정하고 relay·de-jam을 유지함.

`src/harness.py`가 `DefensePolicy` 베이스와 그래프 유틸(adjacency·components·retake_target)을 제공하고,
`src/run_hvt.py`가 시나리오 spec으로 환경을 세워 HVT를 구동함. 상세는 [docs/architecture.md](docs/architecture.md) 참고.

베이스라인 에이전트는 `rule`(휴리스틱 FSM), `llm`(행동 메뉴에서 LLM 선택, 키 없으면 오프라인 stub),
`rl`(tabular Q를 1회 학습 후 고정) 3종임.

## 폴더 구조

```
lab/
├─ src/
│  ├─ run_hvt.py  harness.py          HVT 실험 엔트리 · 방어정책 하네스
│  ├─ run.py sweep.py analyze.py gallery.py make_dataset.py
│  ├─ agents/ (hvt + core)  sim/  viz/  scenarios/ (A01~A21)  configs/
│  └─ counterpart_models/             대안 방어 모델 (비실행 참조)
├─ appendix/ rag_playbook.py          RAG-lite 플레이북 (참조 보관)
├─ docs/     architecture.md demo.md scenarios.md sample_run/
└─ results/ data/                     (생성물, git 제외)
```

## 설치

Python 3.11, CPU. numpy 1.23.5 핀으로 3.12 이상은 미지원.

```bash
python -m venv .venv && source .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
git clone https://github.com/cage-challenge/CybORG
pip install -e ./CybORG --no-deps
pip install -r requirements.txt
```

환경 변수: `ANTHROPIC_API_KEY`(있으면 `llm`이 Claude 호출, 없으면 오프라인 stub),
`SDL_VIDEODRIVER=dummy`(헤드리스 GIF·PNG 저장).

## 실행

```bash
# HVT 방어 실험 (결과: results/hvt_<id>/)
python src/run_hvt.py --scenario A1                          # 단일
python src/run_hvt.py --scenario A11 --recall 0.75 --fp 0.1  # 현실 탐지(오탐·미탐)
python src/run_hvt.py --all                                  # 전체

# 시각화 — 결과 디렉토리를 인자로. docs/sample_run/ 샘플은 CybORG 없이도 동작
python src/viz/dashboard.py docs/sample_run --png   # dashboard.html + 프리뷰 PNG
python src/viz/render.py   docs/sample_run --gif    # 함대 애니메이션 GIF

# 베이스라인 3×3 (rule·llm·rl 공격 3 × 방어 3)
python src/sweep.py src/configs/sweep.yaml          # 첫 실행 시 rl 학습
python src/analyze.py                               # results/ 요약표
```

A1(공급망 웜) 3시드 기준 최종 점령 0.04·가용성 0.88·방어점수 0.94로 웜을 1~2대에서 봉쇄함.
베이스라인 종합점수(5시드, 0~1)는 규칙기반 방어가 최고임.

| 방식 | 공격 A | 방어 D |
|---|---|---|
| rule | 0.53 | 0.81 |
| llm  | 0.52 | 0.58 |
| rl   | 0.47 | 0.43 |

## 결과 시각화

범례: 파랑=아군, 빨강=감염, 삼각형=UAV, 네모=UGV, 보라 링=재밍, 주황 화살표=GPS 스푸핑, 노랑 링=탐지.

HVT 대시보드(A1) — 함대 맵·점수 곡선·전술 로그. 웜 확산을 RetakeSuspicious로 봉쇄(방어 0.94).

![HVT 대시보드](docs/sample_run/dashboard_preview.png)

함대 애니메이션 — HVT가 감염 드론을 탈환하며 확산 억제.

![HVT 함대 애니메이션](docs/sample_run/figs/animation.gif)

행동 단독 검증(`gallery.py`) — red SeizeControl(장악), blue RetakeSuspicious(탈환).

![red SeizeControl](docs/gifs/action_red_SeizeControl.gif)
![blue RetakeSuspicious](docs/gifs/action_blue_RetakeSuspicious.gif)

베이스라인 rule 대 rule 매치업.

![베이스라인 매치업](docs/example_animation.gif)

## 더 보기

[아키텍처](docs/architecture.md) · [데모](docs/demo.md) · [시나리오](docs/scenarios.md) ·
대안 모델 `src/counterpart_models/` · RAG `appendix/rag_playbook.py`
