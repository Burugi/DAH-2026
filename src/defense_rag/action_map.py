"""A5: D3FEND 방어 기법 -> CybORG blue 행동 매핑 (초안).

CybORG blue 행동 공간(agents/actions.py BLUE_CATALOG, id 0-9)은 이미 각 행동에
D3FEND 개념 태그가 달려 있다. 여기서는 그 역방향, 즉 RAG가 찾아낸 D3FEND 기법을
실제로 실행 가능한 blue 행동 id로 되돌리는 표를 정의한다.

매핑은 두 단계:
  1) 기법 개별 override (특정 D3-XXX -> 특정 행동) — 있으면 우선
  2) D3FEND 7대 tactic -> 대표 행동 — 기본 폴백

⚠️ 이 표는 '초안'이다. 실제 CybORG 행동 공간 확정(계획서 Task C2) 후 검증 필요.

blue 행동 id (BLUE_CATALOG, per-step 결정 0-9):
  0 Sleep 1 Monitor 2 Analyse 3 RemoveSessions 4 RetakeSuspicious
  5 RetakeRandom 6 BlockSuspicious 7 AllowTraffic 8 DeployDecoy 9 Failsafe
"""

# blue 행동 메타 (id -> 이름, 대응 D3FEND 개념). actions.py와 일치시킬 것.
BLUE_ACTIONS = {
    0: ("Sleep", "—"),
    1: ("Monitor", "Network Traffic Analysis"),
    2: ("Analyse", "Process/File Analysis"),
    3: ("RemoveSessions", "Process Termination"),
    4: ("RetakeSuspicious", "Re-image / Restore"),
    5: ("RetakeRandom", "Re-image / Restore"),
    6: ("BlockSuspicious", "Network Isolation"),
    7: ("AllowTraffic", "Connectivity Restore"),
    8: ("DeployDecoy", "Decoy / Deception"),
    9: ("Failsafe", "Local Autonomous Defense"),
}

# 1) tactic -> 대표 blue 행동 id (기본 폴백)
TACTIC_TO_ACTION = {
    "Detect":  1,   # Monitor
    "Model":   2,   # Analyse (자산/세션 파악)
    "Isolate": 6,   # BlockSuspicious
    "Evict":   3,   # RemoveSessions
    "Restore": 4,   # RetakeSuspicious
    "Deceive": 8,   # DeployDecoy
    "Harden":  9,   # Failsafe (런타임에 취할 수 있는 가장 가까운 강화 행동)
}

# 2) 특정 D3FEND 기법 override (tactic 기본값보다 더 정확할 때)
TECHNIQUE_OVERRIDE = {
    "D3-PT":   3,   # Process Termination -> RemoveSessions
    "D3-PE":   3,   # Process Eviction
    "D3-NI":   6,   # Network Isolation -> BlockSuspicious
    "D3-ITF":  6,   # Inbound Traffic Filtering
    "D3-OTF":  6,   # Outbound Traffic Filtering
    "D3-NTF":  6,   # Network Traffic Filtering
    "D3-RA":   4,   # Restore Access -> RetakeSuspicious
    "D3-SYSVA": 4,  # System/Restore
    "D3-DE":   8,   # Decoy Environment -> DeployDecoy
    "D3-DP":   8,   # Decoy Public Release / deception
    "D3-DNSTA": 1,  # DNS Traffic Analysis -> Monitor
    "D3-NTA":  1,   # Network Traffic Analysis -> Monitor
    "D3-FA":   2,   # File Analysis -> Analyse
    "D3-PA":   2,   # Process Analysis -> Analyse
}


def d3fend_to_action(d3fend_id, tactic):
    """D3FEND 기법 id + tactic -> (blue_action_id, action_name). 매핑 실패 시 Monitor(1)."""
    if d3fend_id in TECHNIQUE_OVERRIDE:
        aid = TECHNIQUE_OVERRIDE[d3fend_id]
    else:
        aid = TACTIC_TO_ACTION.get(tactic, 1)   # 모르면 일단 관찰(Monitor)
    return aid, BLUE_ACTIONS[aid][0]
