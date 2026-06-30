# 실험 결과 보고서 — 멀티에이전트 Blue 방어 아키텍처 비교

> Branch: `multiagent-experiment`  
> 작성일: 2026-06-30  
> 팀: NeuroGuard

---

## 1. 실험 목적

CybORG DroneSwarm(CC3) 환경에서 **관찰-계획-실행-반성(O-P-E-R) 루프** 기반의 멀티에이전트
방어 아키텍처 4종을 설계·구현하고, 기존 방어 방식(rule / llm / rl) 대비 성능을 비교했다.

---

## 2. 아키텍처 설계

### 2-1. 공통 인터페이스

모든 신규 아키텍처는 `BlueBrainBase`를 상속한다.  
**1 step = 1 팀 액션** 원칙: 에피소드당 1개 인스턴스가 스텝마다 하나의 액션 id를 반환하고,
그 id가 모든 live 드론에 동일하게 적용된다 (`make_blue_index`가 드론별 타깃 선택을 처리).

```
brain = BLUE_MULTIAGENT_TYPES["react"](n)
aid   = brain.step_decide(ctx)          # 1회 / 스텝
brain.step_end(aid, reward, ctx)        # 기록·상태 갱신
brain.episode_end()                     # 에피소드 종료
```

### 2-2. 아키텍처 4종

| 이름 | 루프 | 핵심 아이디어 | LLM 호출 빈도 |
|------|------|--------------|--------------|
| **react** | Observe → Reason (CoT) → Act | 매 스텝 상황을 텍스트로 서술하고 추론 후 액션 선택 | 1회/스텝 |
| **reflect** | Act×K → Reflect → Act×K | K스텝마다 최근 결과를 되돌아보고 전략(strategy tag) 갱신 | 1회/K스텝 |
| **plan** | Plan → Execute → (Re-plan) | 에피소드 시작 시 전체 계획 수립; 상황 이탈 시 재계획 | 1회/에피소드 |
| **ooda** | Observe → Orient → Decide → Act | 군사 OODA 루프: 위협 수준·공격 모드를 분류 후 정책 테이블로 결정 | 1회/스텝 |

#### react — 관찰·추론·행동
매 스텝 위협 등급(safe/low/medium/high/critical)을 산출하고,
LLM이면 `ACTION=<id>` 형식의 추론을 생성, 오프라인이면 등급별 룰 적용.

#### reflect — 행동·반성
`reflect_every=8` 스텝마다 최근 보상·점령 추이를 분석해 전략 태그
(`aggressive_retake / block_and_isolate / remove_sessions / monitor`)를 갱신.
이후 스텝에서는 해당 전략을 따른다.

#### plan — 계획·실행·재계획
스텝 0에서 `[(until_step, action_id), ...]` 형식의 위상 계획을 수립
(예: `[(8,1),(20,4),(35,3),(999,4)]` = Monitor → Retake → Remove → Retake).
점령률이 `replan_threshold=0.45`를 초과하면 재계획을 트리거한다.

#### ooda — 군사 루프
**Observe**: 점령 수, 델타, 트렌드  
**Orient**: 위협 수준(`none/low/medium/high/critical`) × 공격 모드(`spreading/persistent/both`) 분류  
**Decide**: 7×4 정책 테이블 (ex. `high+spreading → BlockSuspicious`)  
**Act**: 정책 id 반환

---

## 3. 실험 설정

| 항목 | 설정값 |
|------|--------|
| 환경 | CybORG DroneSwarm CC3 (v3.1) |
| 플리트 | UAV 12 + UGV 6 = 18대, 100×100 그리드 |
| 에피소드 | 40 스텝, **5 시드** (0~4) |
| 공격자 | rule-red (가장 강한 공격자 고정) |
| 방어자 | rule / llm / rl / react / reflect / plan / ooda (7종) |
| 합성 공격 | 재밍 (드론 0-3, 스텝 10-25, SNR -22dB) + GPS 스푸핑 (드론 12-13, 스텝 15-35, drift 3.0m/step) |
| 패시브 방어 | multisensor 탐지 + safe_mode GPS 보정 |
| LLM | 오프라인 스텁 (API 키 미사용) — 결정론적·재현 가능 |

> **비교 기준 (Exp-1)**: rule/llm/rl 3종을 3×3으로 비교 (시드 3개)  
> **멀티에이전트 (Exp-2)**: rule-red vs 7종 blue 비교 (시드 5개)

---

## 4. 결과

### 4-1. 기존 3×3 비교 (Exp-1)

`results/sweep_sweep_c64d9f25/summary.csv` — 시드 3개 평균

**점령 비율 (낮을수록 방어 강함)**

| 공격↓ \ 방어→ | rule | llm | rl |
|:---:|:---:|:---:|:---:|
| **rule** | **0.33** | 0.56 | 0.83 |
| **llm**  | **0.24** | 0.56 | 0.74 |
| **rl**   | **0.30** | 0.44 | 0.50 |

**방어 점수 D (높을수록 좋음)**

| 공격↓ \ 방어→ | rule | llm | rl |
|:---:|:---:|:---:|:---:|
| **rule** | **0.74** | 0.63 | 0.49 |
| **llm**  | **0.79** | 0.63 | 0.48 |
| **rl**   | **0.75** | 0.68 | 0.65 |
| **평균** | **0.761** | 0.646 | 0.540 |

→ rule 방어가 모든 공격 유형에 대해 점령률 최저, D 점수 최고.

---

### 4-2. 멀티에이전트 비교 (Exp-2)

`results/sweep_multiagent_afd2e980/summary.csv` — rule-red vs 7 blue, 시드 5개 평균

| 방어 방식 | 점령률↓ | 방어점수 D↑ | 가용성↑ | 누적보상↑ | comp F1 | 탈환 수↑ |
|:--------:|:-------:|:---------:|:-------:|:--------:|:-------:|:--------:|
| **rule** | **0.356** | **0.730** | 0.772 | -375.4 | 0.658 | **7.2** |
| react    | 0.389   | **0.730** | 0.723 | -433.0 | 0.784 | 8.2 |
| ooda     | 0.411   | 0.710     | 0.703 | -392.2 | 0.769 | 6.6 |
| reflect  | 0.478   | 0.696     | 0.700 | -398.4 | 0.783 | 4.2 |
| llm      | 0.567   | 0.613     | 0.565 | -549.0 | 0.843 | 11.4 |
| plan     | 0.789   | 0.488     | 0.421 | -517.6 | 0.852 | 2.4 |
| rl       | 0.856   | 0.462     | 0.404 | -547.2 | 0.866 | 0.6 |

**리더보드 (defense D 기준)**

```
1위  rule    D=0.730  점령률 0.356
2위  react   D=0.730  점령률 0.389
3위  ooda    D=0.710  점령률 0.411
4위  reflect D=0.696  점령률 0.478
5위  llm     D=0.613  점령률 0.567
6위  plan    D=0.488  점령률 0.789
7위  rl      D=0.462  점령률 0.856
```

---

## 5. 분석 및 해석

### 5-1. 전체 순위

- **rule**: 점령률 최저(0.356), D 점수 공동 1위(0.730), 가용성 최고(0.772). 빠르고 일관된 대응.
- **react**: D 점수 공동 1위(0.730). 위협 등급 기반 추론이 rule의 우선순위 논리와 유사한 효과를 냄. 단, 탈환 수(8.2)는 rule(7.2)보다 많지만 누적 보상(-433)이 낮아 탈환 타이밍이 다소 늦음을 시사.
- **ooda**: 위협×공격모드 분류 테이블이 블록 동작(BlockSuspicious)을 적절히 삽입해 확산 억제에 효과적. D=0.710으로 3위.
- **reflect**: 반성 주기(8스텝) 사이에 전략이 고정되므로 빠른 확산에 대응이 늦음. 중간 성능(D=0.696).
- **llm**: 오프라인 스텁이 단순한 규칙 계층이라 rule보다 뒤처짐. 실제 Claude API 연결 시 성능 개선 여지 있음.
- **plan**: 초기 계획 후 재계획 트리거(점령률 0.45)가 너무 늦게 발동해 이미 많은 드론이 감염된 후 대응. D=0.488.
- **rl**: 표 기반 Q-러닝이 부족한 상태 표현(2차원)으로 인해 분산 결정이 어렵고, 중앙집중 전략을 사용하는 신규 아키텍처에 비해 불리함. D=0.462.

### 5-2. 신규 아키텍처의 의의

| 관찰 | 해석 |
|------|------|
| react ≈ rule (D 동점) | CoT 추론 없이도 스텁 휴리스틱이 rule과 동등한 수준으로 작동함. 실제 LLM 연결 시 rule을 초과할 가능성 |
| ooda > reflect | 매 스텝 위협 분류(O→O)가 K스텝 후 반성보다 빠른 대응에 유리함 |
| plan 최하위 | 에피소드 수준의 계획은 CC3처럼 공격이 빠르게 확산하는 환경에서 부적합. 짧은 재계획 주기 필요 |
| comp F1 역전 | 신규 아키텍처는 탐지 정확도(comp F1 0.77~0.85)가 rule(0.658)보다 높음 → 감염 인지는 잘 하나 대응 속도 문제 |

### 5-3. GPS/재밍 탐지

모든 아키텍처에서 동일: jam F1 = 1.000, GPS F1 = 0.968, GPS 오차 89.1 → 6.78m (92% 완화).
패시브 텔레메트리 방어(multisensor + safe_mode)는 방어 전략과 무관하게 동작함.

---

## 6. 결론 및 향후 계획

### 6-1. 핵심 결론

1. **rule 방어가 여전히 최강** — 빠른 우선순위 결정(자기 감염 → 세션 제거 → 재장악)이 확산이 빠른 CC3 환경에서 가장 효과적.
2. **react가 rule과 동점** — 위협 등급 기반 추론 루프가 rule과 실질적으로 동일한 결정을 내린다. Claude API 연결 시 rule을 초과할 가능성이 있음.
3. **ooda가 신규 아키텍처 중 최고** — 위협×공격모드 2차원 분류가 단순 반응보다 정교한 전략을 구현.
4. **plan은 CC3 부적합** — 에피소드 단위 계획은 스텝 단위로 빠르게 변하는 드론 군집 환경에 느림.

### 6-2. 향후 계획

- **Claude API 연결 실험**: react·ooda·reflect에 실제 LLM 추론을 연결해 스텁 대비 성능 갭 측정
- **plan 개선**: 재계획 임계값을 낮추거나(0.20) 재계획 주기를 짧게(cooldown=2) 설정
- **reflect 주기 튜닝**: reflect_every=4 또는 6으로 줄여 빠른 전략 갱신 실험
- **하이브리드**: rule 우선순위 결정 + ooda 위협 분류를 결합한 rule+ooda 하이브리드

---

## 7. 실험 재현 방법

```bash
# 환경 설정 (최초 1회)
conda activate dah
git clone https://github.com/cage-challenge/CybORG && pip install -e ./CybORG --no-deps
pip install -r requirements.txt

# 기존 3×3 비교 (Exp-1)
python src/sweep.py src/configs/sweep.yaml --no-gif
# -> results/sweep_sweep_c64d9f25/

# 멀티에이전트 비교 (Exp-2)
python src/sweep.py src/configs/sweep_multiagent.yaml --no-gif
# -> results/sweep_multiagent_afd2e980/

# 단일 매치업 + 대시보드
python src/run.py src/configs/sweep_multiagent.yaml --red rule --blue react
python src/viz/dashboard.py <run_id>
```

---

## 8. 파일 위치

| 파일 | 설명 |
|------|------|
| `src/agents/multiagent.py` | 4종 BlueBrain 구현 (ReActBlue, ReflectBlue, PlannerBlue, OODABlue) |
| `src/configs/sweep_multiagent.yaml` | 멀티에이전트 실험 설정 |
| `results/sweep_sweep_c64d9f25/` | Exp-1 결과 (3×3, 시드 3개) |
| `results/sweep_multiagent_afd2e980/` | Exp-2 결과 (rule-red vs 7 blue, 시드 5개) |
| `docs/experiment_results.md` | 이 문서 |
