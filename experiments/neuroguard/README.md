# NeuroGuard — 공방 실측 연구 (experiments)

팀 NeuroGuard가 CybORG DroneSwarm(CC3) 위에서 돌린 **AI 공격↔방어 자율 공방 실측**. 대회 채점식
`(공격+방어)×가용성`에 맞춘 **가용성 곱셈종합**으로 평가. 구조는 **원본 레포 컨벤션**(configs=YAML 시나리오,
코드 모듈, results/, docs/)에 맞췄다.

> 기존 `src/`·`docs/`·`results/`는 변경하지 않는다. 예외: `src/sim/fleet.py` IndexError 버그픽스 1건.

## 📂 구조
| 경로 | 내용 |
|------|------|
| `configs/attack_scenarios.yaml` | **공격 시나리오 23종 = 단일 진실**(데이터). 원본 `src/configs/scenario_*.yaml` 컨벤션. 시나리오 추가는 여기 한 줄. |
| `runners/` | 실험 드라이버 `exp_*.py`. 핵심 러너는 `attack_scenarios.yaml`을 로드해 사용. |
| `benchmarks/` | 실험 결과 `summary_*.csv` (팀 레포 `.gitignore`의 `results/` 회피용 폴더명). |
| `figures/` | 그림 `fig*.png`, 렌더 애니메이션. |
| `docs/` | 문서(md) — 제출 보고서·시나리오 코드연계·추가연구·마스터 인덱스 등. |

## 🎯 시나리오 (단일 진실)
`configs/attack_scenarios.yaml`이 23종 공격을 sim 기제 파라미터 조합으로 정의한다:
`vectors`(W/J/B 레인) · `inject`(가용성 드레인) · `frag_K`(연결성 분할) · `blackout_p`(위성 단절) ·
`detector_q`(탐지 저하) · `poison_q`(합의 오염) · `tempo`(은밀) · `start_red`(내부자). 러너(`exp_matrix.py`,
`exp_allscenarios.py`)는 이 파일을 로드하므로 **시나리오 정의가 한 곳**에 있다.

## ▶ 재현
```powershell
& <venv>\python.exe runners\exp_matrix.py       # 전 시나리오×방어 매트릭스 → summary_matrix.csv
& <venv>\python.exe runners\exp_allscenarios.py  # 코디네이터 기준 23종 전수
```
- ⚠️ 러너 스크립트는 원 개발환경 **절대경로**(`C:\workspace\...`, 팀 레포 `src`)를 가정하고 출력도 그쪽에 쓴다.
  다른 환경에선 각 러너 상단 경로 상수 조정 필요(향후 파라미터화 예정). `benchmarks/`·`figures/`엔 이미 생성된 산출물이 있다.

## 🔧 포함된 버그픽스 — `src/sim/fleet.py`
`generate_fleet()`에서 공격 `targets` 인덱스가 fleet 크기 `n`을 넘으면 IndexError(소형 fleet). →
`tgt`를 `< n`으로 필터링(빈 경우 skip). 기존 동작(모든 target < n)은 불변. (검증: 소형 fleet 크래시 해소·정상 동작 유지)

## 🎯 핵심 결론 (실측)
1. 위협은 **연결성·가용성**에 집중. 전 23시나리오 코디네이터 곱셈 **0.736**, 통합방어 **0.779**.
2. ML로 통합방어를 깨는 **특이점(웜+대량 허브분할)** 자동 발굴 → 반응형·정적 실패 → **동적 형상 재구성(v2)**으로 해결(0.24→0.74).
3. **탐지 오라클·메트릭 맹점** 두 한계 실측(지속 스텔스=정보이론적 탐지불가→봉쇄, 그레이존 오임무=메트릭 맹점→임무 무결성 축).
4. 명시 설계 > 모방 > from-scratch RL(0.286 실패). 정직한 null(cusum·committed·reserve)도 기록.

> 리뷰 시작점: `docs/시나리오_설명_코드연계_팀공유` · 전체 색인: `docs/00_제출본_인덱스_및_종합요약`.
