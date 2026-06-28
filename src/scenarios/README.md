# Attack Scenarios

공격 시나리오 YAML 파일.

---

## 파일 구조

```
scenarios/
├── __init__.py                   ← 로더
├── A01_supply_chain_...yaml
├── A02_gcs_mavlink_...yaml
│   ...
└── A21_vlm_vla_...yaml
```

---

## YAML 필드 설명

| 필드 | 연결 위치 | 설명 |
|------|----------|------|
| `id` | `__init__.py` | 로더가 파일을 찾는 식별자 |
| `name` | 파일명 | 파일명과 동일하게 유지 |
| `description` | — | 공격 개요 (코드 영향 없음) |
| `mitre_attack` | — | ATT&CK 기법 참조 (코드 영향 없음) |
| `attacks` | `sim/fleet.py` | 물리 공격 정의. `generate_fleet(attacks=...)`에 전달 |
| `red_hint` | `agents/brains.py` | Red 에이전트에 권장하는 `RED_CATALOG` 행동 id |
| `worm` | 미구현 (확장 예약) | 웜 전파 엔진 추가 시 사용할 파라미터 |
| `defense` | `sim/defense.py` | `run_defense()`에 전달되는 탐지·대응 설정 |

### `attacks` 타입

```yaml
- type: gps_spoof
  targets: [0, 1]   # 드론 인덱스 (0~11=UAV, 12~17=UGV)
  t: [start, end]   # 공격 구간 (step)
  drift: 1.2        # 매 step 위치 누적 이탈 (m)

- type: jam
  targets: [0, 1]
  t: [start, end]
  drop: 7.0         # SNR 저하량 (dB). link_up 임계=6 dB
```

### `defense` 필드

```yaml
detector: multisensor   # none | threshold | multisensor
snr_thresh: 4.5         # 재밍 탐지 SNR 임계
gps_thresh: 5.0         # GPS 스푸핑 탐지 IMU 편차 임계 (m)
canary_recall: 0.72     # 감염 드론 탐지율
canary_fp: 0.06         # 정상 드론 오탐률
response: safe_mode     # none | safe_mode | isolate
```

---

## 시나리오 목록

| ID | 파일명 | sim 가능 여부 |
|----|--------|:---:|
| A1  | A01_supply_chain_firmware_trojan_swarm_worm | ✅ |
| A2  | A02_gcs_mavlink_command_injection | — |
| A3  | A03_gps_spoofing_navigation_deception | ✅ |
| A4  | A04_satcom_mitm_replay_attack | — |
| A5  | A05_ros2_dds_middleware_exploit | — |
| A6  | A06_sensor_spoofing_ugv_lidar_camera_imu | ✅ (근사) |
| A7  | A07_rf_jamming_meaconing | ✅ |
| A8  | A08_c2_link_hijack_deauth | ✅ (근사) |
| A9  | A09_ota_firmware_update_exploitation | — |
| A10 | A10_swarm_sybil_byzantine_disruption | — |
| A11 | A11_adversarial_ml_defense_evasion | — |
| A12 | A12_embedded_side_channel_attack | — |
| A13 | A13_pnt_time_sync_attack | ✅ (근사) |
| A14 | A14_multi_domain_coordinated_attack | ✅ |
| A15 | A15_insider_threat_firmware_mission_tampering | — |
| A16 | A16_isr_mission_data_exfiltration | — |
| A17 | A17_swarm_c2_takeover_leader_compromise | — |
| A18 | A18_swarmfuzz_single_node_cascade_collision | ✅ |
| A19 | A19_raven_covert_semantic_navigation_attack | ✅ |
| A20 | A20_incalmo_llm_autonomous_killchain | — |
| A21 | A21_vlm_vla_prompt_injection_cognitive_hijack | — |

> ✅ = `attacks` 블록이 있어 `fleet.py`에 즉시 연결 가능  
> — = 개념 전용 (`attacks: []`), `red_hint` · `defense` 필드만 유효

---

## 사용법

```python
from scenarios import load_scenario, list_scenarios

# 시나리오를 기존 config에 병합
cfg = yaml.safe_load(open("configs/sweep.yaml"))
cfg = load_scenario("A7", cfg)           # attacks + defense 병합

# sim 가능한 시나리오 목록
for s in list_scenarios(sim_only=True):
    print(s["id"], s["name"])
```

```bash
# 단일 실행
python src/run.py configs/sweep.yaml --red rule --blue rule
# (run.py에서 load_scenario를 호출하는 --scenario 인자는 추후 추가 예정)
```
