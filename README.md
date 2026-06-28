# DroneSwarm 공·방 Lab — 3×3 에이전트 비교

CybORG DroneSwarm(CAGE Challenge 3) 위에서 **사이버 공격(red)과 방어(blue)**를 드론 군집에
대해 시뮬레이션하고, **어떤 의사결정 방식이 가장 강한지**를 평가하는 소형 로컬 실험 도구.

## 개요

- 환경: UAV 12 + UGV 6, 2D 격자. 공격·방어 각각 **3가지 방식**을 모두 맞붙여 **3×3 = 9개 매치업**.
- 모든 행동에 **MITRE** 태그(공격 ATT&CK, 방어 D3FEND).
- **핵심 결과: 공격·방어 모두 rule(규칙기반)이 1위**(특히 방어). 아래 종합점수 A/D 참고.

| 방식 | 결정 방법 |
|---|---|
| `rule` | 사람이 짠 휴리스틱 (빠르고 해석 가능, CAGE 우승 레시피) |
| `llm` | LLM이 행동 메뉴에서 선택 (키 없으면 오프라인 stub) |
| `rl` | 학습된 tabular Q 정책 (1회 학습 후 고정) |

```
lab/
├─ README.md  requirements.txt
├─ src/  sweep.py run.py analyze.py make_dataset.py
│        agents/(actions,brains,llm,rl)  sim/(fleet,defense)
│        viz/(plot,render,dashboard,score)  scenarios/(A01~A21)  configs/
├─ docs/  (report, architecture, demo, sample_run)
└─ results/ data/  (생성물, git 제외)
```

## Quick Start

Python 3.11, CPU.

```bash
# CybORG CC3 + 의존성 설치
git clone https://github.com/cage-challenge/CybORG
pip install -e ./CybORG --no-deps
pip install -r requirements.txt

# 전체 3×3 비교 (첫 실행 시 rl 학습, ~1분)
python src/sweep.py src/configs/sweep.yaml      # -> results/sweep_*/
python src/analyze.py                           # 전체 요약표

# 단일 매치업 / MITRE 시나리오 주입
python src/run.py src/configs/sweep.yaml --red rule --blue rl --scenario A1
```

환경변수: `ANTHROPIC_API_KEY`(있으면 llm이 Claude 호출, 없으면 오프라인 stub),
`SDL_VIDEODRIVER=dummy`(헤드리스 GIF 저장).

### Quick test (CybORG 없이)

저장소에 작은 샘플(`docs/sample_run/`)이 포함되어, CybORG 설치나 전체 실행 없이 결과물을 바로
만들 수 있다(numpy·matplotlib·pillow·pygame만 필요).

```bash
python src/viz/dashboard.py docs/sample_run         # -> dashboard.html (브라우저로 열기)
python src/viz/dashboard.py docs/sample_run --png   # -> dashboard_preview.png
python src/viz/plot.py docs/sample_run              # -> figs/*.png
python src/viz/render.py docs/sample_run --gif      # -> figs/animation.gif
```

위 명령으로 만들어지는 결과물 예시:

**대시보드(HTML) 화면 캡처** — 한 화면에 맵 + 전술 로그 + 공격/방어 점수 곡선

![dashboard](docs/sample_run/dashboard_preview.png)

**함대 애니메이션(GIF)**

![sample](docs/sample_run/figs/animation.gif)

## 실험 설정

- **환경**: 18대(UAV 12 + UGV 6), 100×100. 두 채널 — 네트워크(점령·보상)는 CybORG, 위치·통신·
  재밍·GPS는 합성 텔레메트리(정답 라벨 포함).
- **행동**: 매 step 고정 메뉴에서 선택. 공격 ~14종(정찰·익스플로잇·장악·확산·재밍·차단·지속 등),
  방어 ~13종(모니터·분석·세션제거·복구·차단·디코이 + 패시브 탐지). 전체 목록은 `src/agents/actions.py`.
- **시나리오**: MITRE A01~A21을 `src/scenarios/`에 정의. `--scenarios A1,A7` / `sim` / `all`.
- **평가**: 9 매치업 × 5시드 × 40스텝 평균. 진영별 **단일 종합점수**(0~1, 가중치 없는 단순 평균):
  - 공격 A = 평균(점령률, 점령 지속도, 침투 속도, 은밀성)
  - 방어 D = 평균(봉쇄, 웜 탐지, 가용성)

## 에이전트 설계

- **rule**: red = 익스플로잇 성공 시 장악, 아니면 근처 공격(가끔 재밍/지속). blue = 내 드론 감염
  시 세션 제거, 감염 보이면 복구, 아니면 모니터.
- **llm**: 상황 + 행동 메뉴를 프롬프트로 만들어 LLM이 선택. 키 없으면 오프라인 stub(자율 킬체인).
- **rl**: 소형 tabular Q를 학습(rl-red는 rule-blue 상대, rl-blue는 rule-red 상대) 후 고정해 사용.
- 매 step 진영마다 **여러 행동이 동시에** 일어난다(드론·세션마다 1개). 그래서 전술 로그는
  대표(최빈) 행동 + 카운트로 요약한다.

## 결과 시각화

색·모양: **파랑=아군, 빨강=침해** / **삼각형=UAV, 네모=UGV** / 보라 링=재밍, 주황 화살표=GPS
스푸핑, 노랑 링=탐지.

**3×3 종합** (왼쪽 점령률, 가운데 공격점수 A, 오른쪽 방어점수 D)

![grid](docs/gifs/grid_heatmaps.png)

**종합 점수 리더보드 (5시드 평균)**

| 방식 | 공격 A | 방어 D |
|---|---|---|
| **rule** | 0.54 | 0.77 |
| **llm** | 0.52 | 0.64 |
| **rl** | 0.47 | 0.54 |

**공격 우세 예 (rule → rl)** / **방어가 버티는 예 (rule → rule)**

![rule vs rl](docs/gifs/matchup_rule_vs_rl.gif)
![rule vs rule](docs/gifs/matchup_rule_vs_rule.gif)

**행동 단독 검증** — 각 행동이 실제로 동작하고 효과가 다름을 확인 (SeizeControl / RetakeSuspicious)

![seize](docs/gifs/action_red_SeizeControl.gif)
![retake](docs/gifs/action_blue_RetakeSuspicious.gif)

매 매치업은 `dashboard.html`(맵 + 전술 로그 + 점수 곡선)과 GIF·그림을 `results/`에 생성한다(git 제외).

## 더 보기

- 아키텍처 `docs/architecture.md` · 시각화/명령 `docs/demo.md` · 보고서 `docs/report.md`
- 확장: 행동은 `src/agents/actions.py` + `src/agents/brains.py`, 탐지기는 `src/sim/defense.py`,
  시나리오는 `src/scenarios/`.
- 한계: 재밍·GPS는 신호/기하 추상(RF·항법 물리 아님)이라 탐지 점수는 시나리오 설정에 의존한다.
  에이전트 비교 자체는 실제 CybORG 네트워크 채널에서 이뤄진다.
