# 예선보고서 정량화 계획 (이 코드 → DAH2026 예선보고서)

`DAH2026_예선보고서_NeuroGaurd`의 각 절에 **이 lab이 산출하는 수치/표/그림**을 어떻게
채울지의 매핑. 산출물은 `python src/sweep.py src/configs/sweep.yaml` 1회로 전부 생성됨
(`results/sweep_*/summary.csv`, `grid_heatmaps.png`, `grid_curves.png`, 매치업별 `figs/`).

## 배점(100) 대비 핵심 메시지
| 평가영역(배점) | 우리 근거 | 산출물 |
| --- | --- | --- |
| 공격 시나리오 (30) | 공격 3타입 × MITRE 매핑 + 정량 공격지표 | summary.csv, grid_heatmaps(좌), 4.x 표 |
| 방어 전략 (25) | Canary 웜탐지(B5, CC3우승)+멀티센서+가용성보존 대응 | comp_F1/jam_F1/gps, f_defense.png |
| AI 에이전트 (25) | **3×3 모델 비교 실험** + 하이브리드/가용성 설계 | grid_heatmaps/curves, summary.csv |
| 문서 완성도 (10) | 표·그림·GIF·출처(CybORG/KielyM/MITRE) | docs/, animation.gif |

## 절별 매핑

**4.2 시뮬레이션 환경 / Action Space 정의**
- 환경: CybORG DroneSwarm(CC3) + 합성 함대(UAV12+UGV6) — `docs/architecture.md`의 2채널 그림.
- **Action Space 표**: `src/agents/actions.py`의 RED_CATALOG(11) / BLUE_CATALOG(13)을 그대로
  "행동 / sim 프리미티브 / MITRE 태그" 3열 표로. → 보고서가 요구한 "20개 내외 + 근거".
- **CAGE 8개와의 차이**: CC4 blue 8개(Monitor/Analyse/Decoy/Remove/Restore/Block/Allow/Sleep)
  대비, 우리는 드론 군집 도메인(재밍/GPS/웜)에 맞춰 D3FEND 태그로 확장·구체화했음을 명시
  (이데이션 C15 "정해진 행동, 무한한 조합" 논지 인용).

**4.3 공격 시나리오 + MITRE ATT&CK 매핑**
- 시나리오 = `src/configs/sweep.yaml`의 attacks(jam+gps_spoof) + sim 웜.
- 킬체인 표: Discover(T1018) → Exploit(T1210) → Seize/Persist(T1078/T1542) → SpreadWorm(T1021)
  → Jam(T1498/T1499)/BlockComms(T1565). (이데이션 A1 대표 + A3 + A7 + A14 복합)

**4.4 공격 에이전트 구현**
- `src/agents/brains.py`의 `RuleRed/LLMRed/RLRed` 코드구조 요약 + 각 정책 1단락.
- 정량: summary.csv의 공격지표(`final_compromise`, `time_to_first_compromise`,
  `compromise_auc`)를 red 타입별로 비교 → "어느 공격이 더 빠르고 넓게 침투하는가".

**5.3 하이브리드 방어 / 5.4 Action↔정책 매핑**
- 규칙기반 방어 = `blue_decide("rule")` 우선순위(자기감염→Remove, 의심→Retake, else Monitor) +
  Canary 웜탐지(`sim/defense.py`, B5 CC3우승) + 멀티센서 GPS + safe_mode(가용성 보존, B7).
- 정량: `comp_F1`(웜탐지), `jam_F1`, `gps_F1`, `gps_err_before→after`(완화) → f_defense.png.
- ⚠️ 솔직한 표기: 현재 rule/llm/rl은 **개별** 에이전트. 이데이션 C3의 "규칙+RL+LLM 단일
  하이브리드 코어"는 향후 과제로 명시(7.4 로드맵).

**6.2 벤치마크/지표**
- CybORG CC3 채택 근거 + reward(=blue 평균 보상)·Episode 지표 정의. CybORG 시각화는 자체
  pygame 뷰어/animation.gif로 대체(CC3 native 렌더 없음) — 출처·이유 명시.

**6.3 모델별 성능 비교 실험 ★핵심**
- **이 lab의 3×3 sweep이 곧 모델 비교 실험.** 매핑: rule→M1 Heuristic, llm→LLM-프롬프팅,
  rl→경량 RL(tabular Q). 보고서의 PPO/DQN/GNN/하이브리드는 **향후 확장**으로 표기.
- 표: summary.csv → (red×blue) 9행, mean±std는 meta.json. 그림: `grid_heatmaps.png`
  (final_compromise / blue_reward / comp_F1), `grid_curves.png`(감염수-step ×9).
- 강·약점 분석: rule-blue가 모든 공격을 최저 감염으로 억제(예: rule열 final_comp 0.17~0.33)
  → **KielyM(CC4) "휴리스틱 > MARL" 결론과 정합**(직접 인용).

**6.4 최종 모델 선정**
- 누적보상·감염억제·comp_F1 종합 → rule(또는 rule+RL 하이브리드) 선정 근거를 summary.csv 수치로.

**7. 결론/한계**
- 한계: jam/GPS는 합성 추상(RF/항법 물리 X) → 탐지 F1이 설정의존, sim-to-real gap,
  action-space 이식성. 7.4: PPO/DQN/GNN·단일 하이브리드 코어·미지공격 4계층(B20) 로드맵.

## 만들어야 할 그림/표 체크리스트
- [표1] Action Space (red11/blue13 + MITRE) ← actions.py
- [표2] 9개 매치업 정량 비교 ← summary.csv
- [그림1] 시스템 아키텍처 ← docs/architecture.md
- [그림2] 3×3 히트맵 ← grid_heatmaps.png
- [그림3] 감염-step 곡선(9패널) ← grid_curves.png
- [그림4] 방어 탐지/완화 ← f_defense.png
- [그림5] step별 시각화 ← animation.gif (데모영상 docs/demo.md)

## 캡션에 넣을 출처
CybORG/CAGE CC3 (Standen 2021), KielyM CC4 (AAAI-25), MITRE ATT&CK/D3FEND.
