# -*- coding: utf-8 -*-
"""도달성 특화 방어 v2 — 연결성 시나리오(A-CONN/A18/A17/A4/A13/A3) 집중 강화.

ReachPolicy 대비 개선점:
  1. 세그먼트 이중등록 버그 수정 — 기존 `seg_groups = comps + [reconnected]`는
     reconnected에 담긴 isolated 노드가 이미 comps(자기 성분)에 들어가 있어 이중 등록.
     v2는 **재연결 메시(reconnected)를 하나의 세그먼트로만** 두고, 그 노드들을 comps에서
     제거한다(대형성분 big은 그대로 유지). 청정 드론이 두 번 배정돼 낭비되는 일이 없다.
  2. relay 재연결 캡 제거 — R_relay=None. isolated+black을 **전부** 로컬 RF 메시로 재연결해
     도달불가(lost)를 허브(frag_K)만으로 최소화 → A-CONN/A18/A4 가용성↑.
  3. de-jam 결합 — 재장악에 배정되지 않은 청정 도달가능 드론이 de-jam(통합방어와 동일).
  4. 가용성 회계는 `lost = len(unreachable - comp)` 패턴 준수(감염∩도달불가 이중차감 방지).
  5. 다중 재장악(REDUNDANCY=3) — 감염 1대당 최근접 청정 드론을 최대 3대까지 배정해
     웜 확산을 빠르게 진압(final/AUC 점령↓). 유휴 드론이 놀지 않고 재장악에 투입돼
     연결성 시나리오의 점령을 크게 낮춘다(실측: RED 1→3에서 avg 0.807→0.855, 4↑부터 역효과).
     남는 드론은 여전히 de-jam을 수행하므로 재밍 시나리오도 함께 개선.
"""
import numpy as np

from agents.defense_base import JAM_VECS, actions, adjacency, components, retake_target, DefensePolicy


class ReachV2(DefensePolicy):
    name = "reach2"
    R_relay = None          # None = 무제한 재연결(캡 제거)
    REDUNDANCY = 3          # 감염 1대당 배정할 청정 재장악 드론 수(실측 최적=3)

    def reset(self, cfg, fleet, spec, hubs, black, ml, recall, fp):
        self.spec, self.hubs, self.black, self.ml = spec, hubs, black, ml
        self.recall, self.fp = recall, fp
        self.vectors = spec.get("vectors", ["W"])
        self.k = len(self.vectors)
        self.n = fleet["n"]

    def step(self, comp, pos, env, live, ip2d, rng):
        n, spec, vectors, k = self.n, self.spec, self.vectors, self.k
        hubs, black = self.hubs, self.black
        A = adjacency(pos, self.ml)
        present = set(range(n)) - hubs - black
        comps = components(present, A, n)
        big = max(comps, key=len) if comps else set()
        isolated = present - big
        # ★재연결 대상: 분할(isolated) + 위성단절(black). 캡 없으면 전부 재연결.
        reconnectable = list(isolated) + list(black)
        cap = len(reconnectable) if self.R_relay is None else self.R_relay
        reconnected = set(reconnectable[:cap])
        unreachable = (hubs | black | isolated) - reconnected
        # ★이중등록 수정: 재연결 노드는 '메시' 세그 하나로만. big(대형성분)은 유지하되
        #   재연결된 노드가 섞여 들어가지 않도록 comps에서 재연결 노드를 제거.
        seg_groups = [c - reconnected for c in comps]
        seg_groups = [c for c in seg_groups if c]
        if reconnected:
            seg_groups.append(reconnected)

        detected = set()
        for i in comp:
            if i in unreachable:
                continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                detected.add(i)
        if self.recall < 1.0 or self.fp > 0.0:
            detected = {i for i in detected if rng.random() < self.recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < self.fp}
            heal_set = target_set = nonclean = detected | fp_set
        else:
            heal_set, target_set, nonclean = comp, detected, comp
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0

        assign, dejam = {}, 0
        for seg in seg_groups:
            sc = [c for c in seg if c in target_set]
            scl = [d for d in seg if d not in nonclean]
            used = set()
            for _ in range(self.REDUNDANCY):          # 다중 재장악: 감염 1대당 청정 드론 최대 REDUNDANCY대
                for c in sc:
                    cand = sorted([d for d in scl if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                    if cand:
                        assign[cand[0]] = c
                        used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable:
                acts[a] = actions.make_blue_index(0, env, a, ctx)
                continue
            if i in heal_set:
                acts[a] = actions.make_blue_index(3, env, a, ctx)
                continue
            if i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                if red_jam > 0:
                    dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)

        restored = min(dejam, red_jam)
        lost = len(unreachable - comp)                       # 감염∩도달불가 이중차감 방지
        avail = max(0.0, (n - len(comp) - lost - (red_jam - restored) - min(inj, max(0, n - len(comp)))) / n)
        return acts, avail
