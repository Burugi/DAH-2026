[구현] DAH2026 예선 실험 — Action 제한 + 모델 비교

## 배경/합의

코드팀 확정 4가지: ① 벤치마크는 **CybORG 표준**을 그대로 쓴다(LIG 채점식 미채택, 환경 대수정 회피). ② Action은 CybORG 기존 집합에서 **소수만 제한 채택**(자체 20개 발명 금지). ③ 모델은 **규칙기반 휴리스틱 + RL 하이브리드**로 간다. ④ 결론은 정해진 action 내에서 모델을 비교해 **최고 모델을 본선**에 투입한다. 팀 레포 `Burugi/DAH-2026`(CybORG CC3/DroneSwarm 기반)을 확장하며, 보상·시각화·평가는 기존 것을 재사용한다.

## 구현 대상 (시나리오 S1~S4)

- [ ] **S1. 웜 확산(A1)** — `red_class: RedDroneWorm` (실제 sim) / config: baseline, defended
- [ ] **S2. GPS 스푸핑(A3)** — `attacks: gps_spoof` (fleet.py 합성) / config: defended
- [ ] **S3. 재밍(A7)** — `attacks: jam` (fleet.py 합성) / config: fsm_red
- [ ] **S4. 복합(웜+재밍+GPS)** — combined config / config: combined
- [ ] (red 변형) **FSM red** — `red_class: DroneRedAgent` / config: fsm_red

## 모델 비교 (정해진 action 내 성능 비교)

- [ ] **M1. 휴리스틱 baseline** — 규칙(react/remove/retake) + **Canary/Whistle** 추가. 위치: `agents.py`의 `blue_action()`에 `"canary"` 분기 추가 (하트비트 누락 + snr 정상 → 감염 의심 → RetakeControl/RemoveOtherSessions).
- [ ] **M2. PPO(RL)** — Stable-Baselines3 PPO. 위치: PettingZoo 래퍼(`PettingZooParallelWrapper`) 위 신규 `train_ppo.py`. 보상은 CybORG 기본 사용.
- [ ] **M3. 하이브리드 (주력 ★★)** — 휴리스틱을 action-mask로, 나머지를 PPO가 결정. 위치: M1을 마스킹/override 레이어로, M2 PPO 위에 적용.
- [ ] **M4. GNN+PPO (옵션)** — 네트워크 그래프 표현으로 범용성 어필. 위치: PyTorch Geometric (여유 시).

## 평가지표 (전부 CybORG/레포 재사용)

- **reward** — CybORG 누적 보상
- **red_owned** — 감염 드론 수
- **F1** — 방어 탐지 F1 (jam / gps)
- **gps_err** — GPS 오차 before → after
- **가용성 유지율** — 본선 채점이 가용성 곱셈형이므로 가용성 보존 관점 병기

> 산출 경로: `run.py` → `results/<run_id>/`(log.csv·arrays.npz·meta.json[F1]), `analyze.py` → `summary.csv`, `plot.py`/`render.py` → 그래프·시각화.

## 작업 분담 (구현가이드 §7 기반)

- [ ] (즉시) M1 휴리스틱 강화(Canary/Whistle) + S1~S4 config로 baseline 성능 측정
- [ ] (병행) M2 PPO 학습 파이프라인(SB3) 구축
- [ ] (통합) M3 하이브리드 → 성능 비교표 작성
- [ ] (여유) M4 GNN(범용성 어필)
- [ ] 각 단계 결과를 `analyze.py`의 `summary.csv`로 모아 논문 §결과에 직접 투입

## 범위 명시

A2/A4/A5/A6(MAVLink·SATCOM·ROS2·센서)는 **CybORG 미모델 → 보고서 §위협모델에서 개념적 위협으로만 서술하고 구현하지 않는다.** 실험은 **S1~S4**로 한정한다.

---

상세는 `코드팀_구현가이드.md` 참조.
