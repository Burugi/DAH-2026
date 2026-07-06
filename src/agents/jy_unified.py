# jy_unified — NeuroGuard 방어모델 구조 (원본: neuroguard/agents/heuristic.py)
# MatrixHeuristic('ultimate') — +relay+de-jam  · 실동작 통합은 본선 구조에서.
# -*- coding: utf-8 -*-
"""휴리스틱 방어 두뇌 — 무방어 / 코디네이터 / 통합방어.

`experiments/neuroguard/runners/exp_matrix.py`의 none/coord/ultimate 롤아웃 로직을
**그대로 이식**했다(수치 재현 목적). 세 모드가 토폴로지·탐지·가용성 회계를 공유하므로
하나의 클래스에서 self.mode로 분기한다.
"""
import numpy as np

from neuroguard.agents.base import DefensePolicy
from neuroguard.harness import JAM_VECS, actions, adjacency, components, retake_target


class MatrixHeuristic(DefensePolicy):
    """mode ∈ {none(무방어), coord(코디네이터), ultimate(통합방어)}."""

    def __init__(self, mode, R_relay=4, ch2="monitor"):
        assert mode in ("none", "coord", "ultimate")
        assert ch2 in ("real", "monitor")
        self.mode = mode
        self.name = {"none": "무방어", "coord": "코디네이터", "ultimate": "통합방어"}[mode]
        self.R_relay = R_relay
        # 채널②(재밍/스푸핑 대응) 정책. 기존 '가짜' de-jam/relay 자가신고 회계는 제거됨(step 참조).
        #   real    — 재밍 유휴 드론에 진짜 Failsafe(9)=AllowTraffic, 스푸핑에 SafeMode(12) 발행(팀 액션셋).
        #             ★실측(sweep jam+gps, seeds0-4): 물리 가용성 0.872→0.824(-0.047 악화). 이유:
        #             (1) link_up=(snr>6)은 build_env 사전계산 정적 물리량 → 어떤 블루 액션도 재밍 복구 불가,
        #             (2) AllowTraffic이 차단 트래픽을 재허용 → 웜/red 재침투로 점령↑,
        #             (3) SafeMode(12)는 make_blue_index서 미처리 → Sleep(무효).
        #   monitor — do-no-harm 기본값: 유휴 드론 관측만. 물리 가용성 0.872(coordinator와 동등, 가짜 회계 제거).
        self.ch2_mode = ch2

    def reset(self, cfg, fleet, spec, hubs, black, ml, recall, fp):
        self.spec = spec
        self.hubs, self.black, self.ml = hubs, black, ml
        self.recall, self.fp = recall, fp
        self.vectors = spec.get("vectors", ["W"])
        self.k = len(self.vectors)
        self.n = fleet["n"]
        # 채널② 진짜 액션용 실측 텔레메트리(link_up: snr>6 정적 물리량, label_gps: GPS 스푸핑 라벨).
        # matchup은 spec={}로 reset하므로 vectors 기반 red_jam 프록시가 0 → 실측 텔레메트리로 재밍/스푸핑 탐지.
        self.fleet = fleet
        self._t = 0
        self.ch2 = {"failsafe": 0, "safemode": 0, "monitor": 0}   # 채널② 발행 액션 계측(정직 보고용)

    def step(self, comp, pos, env, live, ip2d, rng):
        n, spec, vectors, k = self.n, self.spec, self.vectors, self.k
        mode, hubs, black = self.mode, self.hubs, self.black
        A = adjacency(pos, self.ml)
        present = set(range(n)) - hubs - black
        comps = components(present, A, n)
        big = max(comps, key=len) if comps else set()
        isolated = present - big
        reconnected = set(list(isolated)[:self.R_relay]) if mode == "ultimate" else set()
        iso2 = isolated - reconnected
        if mode == "ultimate":
            seg_groups, unreachable = comps, hubs | black
        else:
            seg_groups, unreachable = [big | reconnected], hubs | black | iso2

        # 탐지: detector_q 저하 · poison_q 오염 → 코디네이터가 '보는' 감염
        detected = set()
        for i in comp:
            if i in unreachable:
                continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                detected.add(i)
        # 현실 탐지: recall(FN) + fp(오탐). 결정은 believed, 채점은 true comp
        realistic = (self.recall < 1.0 or self.fp > 0.0)
        if realistic:
            detected = {i for i in detected if rng.random() < self.recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < self.fp}
            believed = detected | fp_set
            heal_set = target_set = nonclean = believed
        else:
            heal_set, target_set, nonclean = comp, detected, comp

        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0

        # ── 채널② 진짜 액션용: 현재 스텝의 실측 재밍(link_up==0)·GPS 스푸핑(label_gps) 탐지 ──
        t_idx = min(self._t, len(self.fleet["link_up"]) - 1)
        link_up_t = self.fleet["link_up"][t_idx]
        _lg = self.fleet.get("label_gps")
        gps_t = _lg[t_idx] if _lg is not None else None
        self._t += 1

        acts = {}
        if mode == "none":
            for a in live:
                acts[a] = actions.make_blue_index(1, env, a, ctx)          # 전원 관측(무방어)
        else:
            assign = {}
            for seg in seg_groups:                                          # 세그먼트별 최근접 청정 드론 1대 배정
                sc = [c for c in seg if c in target_set]
                scl = [d for d in seg if d not in nonclean]
                used = set()
                for c in sc:
                    cand = sorted([d for d in scl if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                    if cand:
                        assign[cand[0]] = c
                        used.add(cand[0])
            for a in live:
                i = int(a.split("_")[-1])
                if i in unreachable:
                    acts[a] = actions.make_blue_index(0, env, a, ctx)
                    continue
                if i in heal_set:
                    acts[a] = actions.make_blue_index(3, env, a, ctx)       # 자가복구
                    continue
                if i in assign:
                    acts[a] = retake_target(env, a, assign[i], ip2d, sleep)  # 재장악
                elif mode == "ultimate" and self.ch2_mode == "real" and gps_t is not None and gps_t[i]:
                    acts[a] = actions.make_blue_index(12, env, a, ctx)       # SafeMode(12) — GPS 스푸핑 교정(진짜 액션)
                    self.ch2["safemode"] += 1
                elif mode == "ultimate" and self.ch2_mode == "real" and link_up_t[i] == 0:
                    acts[a] = actions.make_blue_index(9, env, a, ctx)        # Failsafe(9) — 자율 링크복구/세션제거(진짜 액션)
                    self.ch2["failsafe"] += 1
                else:
                    acts[a] = actions.make_blue_index(1, env, a, ctx)        # Monitor(관측)
                    if mode == "ultimate":
                        self.ch2["monitor"] += 1

        # 자가신고 부풀림 제거(정직 회계): 진짜 액션(Failsafe/AllowTraffic/SafeMode)은
        #   정적 물리량 link_up을 못 바꾼다 → 재밍 '복구(restored)' 자가가정 삭제.
        #   relay 회계(reconnected)도 삭제 → 실제 고립은 isolated 전체로 계상(iso2 아님).
        lost = len((hubs | black | isolated) - comp)        # 감염∩도달불가 이중차감 방지(감염은 len(comp)서 이미 차감)
        avail = max(0.0, (n - len(comp) - lost - red_jam - min(inj, max(0, n - len(comp)))) / n)
        return acts, avail
