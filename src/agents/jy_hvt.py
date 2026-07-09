# -*- coding: utf-8 -*-
"""HVT — Hypothesis-Verify-Trigger 방어.

belief 상위를 무조건 재장악하지 않고, 각 후보의 재장악 실익을 world-model
반사실 시뮬로 검증해 트리거를 넘는 것만 파괴적으로 재장악한다.

  ① 가설: belief b[i]=P(감염)를 웜확산 전진예측 + 탐지 융합으로 유지 (detected+인접만 사용).
  ② 검증: 후보 i에 대해 Δ_i = "지금 i를 재장악하면 앞으로 H스텝 막는 확산량".
  ③ 트리거: Δ_i > τ 인 타겟만 파괴적 재장악(RetakeControl). 나머지는 비파괴 자가치유로 hold.
  ④ 실행: 재장악을 Δ 큰 순으로 다중배정(REDUNDANCY), relay/de-jam 유지.

belief·Δ·target 계산에는 관측(detected)+인접(A)만 쓴다. true 감염(comp)은 탐지 샘플링과
가용성 회계에만 사용. recall=1·fp=0이면 검증 루프를 끄고 파괴적 재장악을 전량 수행한다.
"""
import numpy as np

from agents.defense_base import DefensePolicy, JAM_VECS, adjacency, components, retake_target
from agents import actions


class HVTDefense(DefensePolicy):
    name = "hvt"

    # --- 응답 메커니즘 ---
    R_relay = None            # 무제한 relay 재연결
    REDUNDANCY = 3            # 재장악 확정 타겟 1대당 청정 재장악 드론 수

    # --- ① 가설(belief) 하이퍼 ---
    BETA0 = 0.35              # 웜확산 전진모델 감염률
    DET_BELIEF = 0.6          # 탐지=증거 belief 하한 (나머지는 코로보레이션이 부양)
    CORR_GAIN = 0.35          # 확산예측 지지에 따른 detected belief 추가부양
    MISS_KEEP = 0.65          # 미탐지 노드 belief 유지율 (미탐지 ≠ 청정)
    RETAKE_DECAY = 0.2        # 지난 스텝 재장악한 노드 belief 감쇠
    FRONTIER_THRESH = 1       # frontier 편입: detected 이웃 수 ≥ THRESH → 즉시 비파괴 heal
    MISS_TH = 0.4             # belief≥MISS_TH 미탐지 노드 = 예측감염(multi-hop) → 비파괴 자가치유

    # --- ②③ 검증·트리거 하이퍼 ---
    HORIZON = 4               # 반사실 시뮬 호라이즌 H
    TAU = 1.6                 # 트리거 임계 τ (Δ>τ 만 파괴적 재장악)
    SORT_ORDER = False        # 재장악 배정: Δ순(True) / 최근접 우선(False, de-jam 배정 보존)

    # --- 피드백(BETA 온라인 보정) ---
    BETA_LR = 0.06
    BETA_MIN, BETA_MAX = 0.15, 0.55

    def reset(self, cfg, fleet, spec, hubs, black, ml, recall, fp):
        self.spec, self.hubs, self.black, self.ml = spec, hubs, black, ml
        self.recall, self.fp = recall, fp
        self.vectors = spec.get("vectors", ["W"])
        self.k = len(self.vectors)
        self.n = fleet["n"]
        self.FRONTIER_THRESH = getattr(self, "FRONTIER_THRESH", 1)
        self.belief = np.zeros(self.n)          # 리셋 시 0(감염 사전지식 없음). detected로만 seed.
        self.beta = self.BETA0
        self.prev_retaken = set()
        self.prev_expected = 0.0

    # world-model: 웜확산 전진 (belief 전파)
    def _spread(self, b, A, beta):
        """b_next[i] = b[i] + (1-b[i])*(1 - Π_{A[i,j]}(1-beta*b[j])). comp 미사용, b·A만."""
        L = np.log(1.0 - beta * b)              # b<=DET/spread<1, beta<1 → 1-beta*b>0
        prod = np.exp(A.astype(np.float64) @ L)  # Π over neighbors
        return b + (1.0 - b) * (1.0 - prod)

    def _sim_total(self, b0, A, beta, H, zero=None):
        """belief b0를 H스텝 전진하며 누적점령(Σ b) 반환. zero!=None이면 그 노드를 매스텝 0 고정
        (=지금 재장악해 청정 유지). 반사실 비교용."""
        b = b0.copy()
        if zero is not None:
            b[zero] = 0.0
        tot = 0.0
        for _ in range(H):
            b = self._spread(b, A, beta)
            if zero is not None:
                b[zero] = 0.0
            tot += float(b.sum())
        return tot

    def step(self, comp, pos, env, live, ip2d, rng):
        n, spec, vectors, k = self.n, self.spec, self.vectors, self.k
        hubs, black = self.hubs, self.black
        A = adjacency(pos, self.ml)
        present = set(range(n)) - hubs - black
        comps = components(present, A, n)
        big = max(comps, key=len) if comps else set()
        isolated = present - big
        reconnectable = list(isolated) + list(black)
        cap = len(reconnectable) if self.R_relay is None else self.R_relay
        reconnected = set(reconnectable[:cap])
        unreachable = (hubs | black | isolated) - reconnected
        seg_groups = [c - reconnected for c in comps]
        seg_groups = [c for c in seg_groups if c]
        if reconnected:
            seg_groups.append(reconnected)

        # 관측: detector가 true comp를 detector_q 확률로 샘플
        detected = set()
        for i in comp:
            if i in unreachable:
                continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                detected.add(i)

        realistic = self.recall < 1.0 or self.fp > 0.0
        missed = set()
        order = {}

        if not realistic:
            # 완전관측 분기: belief/검증 루프 OFF, 탐지된 감염 전량 재장악
            heal_set = target_set = nonclean = comp
            order = {i: 0.0 for i in comp}
        else:
            detected = {i for i in detected if rng.random() < self.recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < self.fp}
            evidence = detected | fp_set

            # ⑤ 피드백: 지난 예측 vs 이번 실제 detected → BETA 보정
            actual = float(len(detected))
            if self.prev_expected > 0 or actual > 0:
                if actual > self.prev_expected + 1e-9:
                    self.beta = min(self.BETA_MAX, self.beta + self.BETA_LR)
                elif actual < self.prev_expected - 1e-9:
                    self.beta = max(self.BETA_MIN, self.beta - self.BETA_LR)

            # ① 가설: belief 웜확산 전진예측 + detected 융합(코로보레이션 가중)
            b_pred = self._spread(self.belief, A, self.beta)
            new_b = np.empty(n)
            for i in range(n):
                if i in evidence:
                    # 탐지=증거 하한, 확산예측(이웃 belief 지지)만큼 추가 부양 → 고립 단발오탐은 낮게 유지
                    new_b[i] = min(0.98, max(self.DET_BELIEF, b_pred[i]) + self.CORR_GAIN * b_pred[i])
                else:
                    new_b[i] = b_pred[i] * self.MISS_KEEP           # 미탐지 ≠ 청정
                if i in self.prev_retaken:
                    new_b[i] *= self.RETAKE_DECAY                   # 되찾음 = 신념↓
            self.belief = new_b

            # frontier: detected의 청정 이웃(A·detected만). recall<1로 놓친 이웃 감염을
            #   1-hop 선제 비파괴 자가치유로 catch.
            deg = {}
            for i in detected:
                row = A[i]
                for j in range(n):
                    if row[j] and j not in evidence and j not in unreachable:
                        deg[j] = deg.get(j, 0) + 1
            frontier = {j for j, c in deg.items() if c >= self.FRONTIER_THRESH}
            # belief 예측감염: 누적 belief가 높은 미탐지 노드(multi-hop).
            #   1-hop frontier가 못 잡는 깊은/저탐지(A10 Sybil·A12 stealth) 놓친 감염을
            #   비파괴 자가치유(RemoveOtherSessions)로 청소. 청정이면 무해.
            missed_belief = {
                i for i in range(n)
                if i not in unreachable and i not in evidence and self.belief[i] >= self.MISS_TH
            }
            missed = frontier | missed_belief          # 유휴 드론 비파괴 자가치유 대상

            # ② 검증: 후보(=evidence)에 대해서만 반사실 Δ.
            #   Δ_i = i를 지금 재장악하면 앞으로 H스텝 막는 점령량(자신+하류확산).
            candidates = [i for i in evidence if i not in unreachable]
            base_tot = self._sim_total(self.belief, A, self.beta, self.HORIZON)
            deltas = {}
            for i in candidates:
                int_tot = self._sim_total(self.belief, A, self.beta, self.HORIZON, zero=i)
                deltas[i] = base_tot - int_tot

            # ③ 트리거: Δ>τ 통과분만 파괴적 재장악. 미통과(고립오탐·확산정지)는
            #   heal_set(evidence)에 남아 비파괴 자가치유로 hold — 파괴적 Restore 낭비 회피.
            target_set = {i for i in candidates if deltas[i] > self.TAU}
            order = {i: -deltas[i] for i in target_set}  # Δ 큰 순 배정

            heal_set = evidence                          # 증거 전부 비파괴 자가치유
            nonclean = evidence                          # 재장악 드론 풀 제외(frontier는 풀에 유지)

        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0

        # ④ 실행: 세그먼트별 다중 재장악 배정(최근접 우선; SORT_ORDER=True면 Δ순)
        assign, dejam = {}, 0
        sort_order = getattr(self, "SORT_ORDER", False)
        for seg in seg_groups:
            prim = [c for c in seg if c in target_set]
            if sort_order:
                prim = sorted(prim, key=lambda c: order.get(c, 0.0))
            scl = [d for d in seg if d not in nonclean]
            used = set()
            for _ in range(self.REDUNDANCY):
                for c in prim:
                    cand = sorted(
                        [d for d in scl if d not in used and d != c],
                        key=lambda d: np.linalg.norm(pos[d] - pos[c]),
                    )
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
                acts[a] = actions.make_blue_index(3, env, a, ctx)    # 증거 자가치유(비파괴)
                continue
            if i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)   # 검증통과 타겟 파괴적 재장악
            elif realistic and i in missed:
                acts[a] = actions.make_blue_index(3, env, a, ctx)    # 유휴 frontier 비파괴 자가치유(놓친 이웃감염 catch)
            else:
                if red_jam > 0:
                    dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)    # de-jam / monitor

        # ⑤ 피드백 상태 저장
        if realistic:
            self.prev_retaken = set(assign.values())
            reach_mask = np.array([1.0 if i not in unreachable else 0.0 for i in range(n)])
            self.prev_expected = float(self.recall * np.sum(b_pred * reach_mask))

        restored = min(dejam, red_jam)
        lost = len(unreachable - comp)                       # 감염∩도달불가 이중차감 방지
        avail = max(0.0, (n - len(comp) - lost - (red_jam - restored) - min(inj, max(0, n - len(comp)))) / n)
        return acts, avail
