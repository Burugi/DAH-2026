# jy_blade — NeuroGuard 방어모델 구조 (원본: neuroguard/agents/blade.py)
# BladeDefense — belief 예측 폐루프  · 실동작 통합은 본선 구조에서.
# -*- coding: utf-8 -*-
"""BLADE — Belief-Loop Adaptive Defense (닫힌 루프 예측 방어).

기존 모든 방어(none/unified/reach*/mpc)는 **단방향**이다: 관측(detected) → 결정.
현실 탐지(recall<1, fp>0)에서 웜이 이웃으로 퍼졌는데 탐지가 놓친 감염은
reach3가 1-hop heuristic frontier(detected의 청정 이웃)로만 근사한다.

BLADE는 **닫힌 루프**다: belief state + world-model 확산 시뮬 + 피드백.
  ┌─ ① 예측 ─ 웜확산 전진모델로 belief를 전파(미탐지 감염을 원리적으로 예측)
  │  ② 관측·융합 ─ detected는 강한 증거(→0.9), 미탐지는 청정 아님(*MISS_KEEP 유지)
  │  ③ 플래닝 ─ belief 랭킹(감염추정) + 확산위험으로 재장악/자가치유 대상 선정
  │  ④ 실행 ─ reach2식 다중배정·relay·de-jam (응답 메커니즘 재사용)
  └─ ⑤ 피드백 ─ 실제 detected vs 예측 비교로 BETA 온라인 보정(tempo/다벡터 자동적응)

reach3와의 본질 차이:
  * reach3 frontier = {detected의 청정 이웃, 이웃수≥1} — **1-hop, 무기억, 고정**.
  * BLADE belief = 확산모델로 **multi-hop 전파 + 스텝간 누적 + BETA 자기보정**.
    → 탐지가 깊게 놓친 감염 사슬을 belief가 예측해 선제 청소한다.

★오라클 누수 없음(코드 보장):
  belief 전파·플래닝은 **오직 detected(관측) + A(인접행렬)** 로만 이뤄진다.
  step의 `comp`(true 감염)는 (a) detector 관측 샘플링(detector_q·recall·fp — 모든
  기존 정책과 동일한 관측 채널)과 (b) 가용성 회계(red_jam/inj/lost) 에만 쓰인다.
  belief[]·heal_set·target_set·nonclean 어디에도 `comp`가 직접 들어가지 않는다.
  오라클 분기(recall=1,fp=0)는 belief 루프를 끄고 reach2와 동일하게 동작한다.
"""
import numpy as np

from neuroguard.agents.base import DefensePolicy
from neuroguard.harness import JAM_VECS, actions, adjacency, components, retake_target


class BladeDefense(DefensePolicy):
    name = "blade"

    # --- 응답 메커니즘(reach2/reach3 계승) ---
    R_relay = None            # 무제한 relay 재연결
    REDUNDANCY = 3            # 감염추정 1대당 청정 재장악 드론 수

    # --- belief 루프 하이퍼(튜닝 대상) ---
    BETA0 = 0.35              # 웜확산 전진모델 초기 감염률
    MISS_KEEP = 0.7           # 미탐지 노드 belief 유지율(미탐지 ≠ 청정)
    RETAKE_DECAY = 0.2        # 지난 스텝 재장악한 노드 belief 감쇠
    DET_BELIEF = 0.9          # 탐지=강한 증거 하한
    MISS_TH = 0.35            # 이 belief 이상의 미탐지 노드 = 예측 감염 → 선제 자가치유
    USE_FP_IN_BELIEF = True   # fp도 detector 증거로 belief 부양(현실적: 오탐인지 모름)

    # --- 피드백(BETA 온라인 보정) ---
    BETA_LR = 0.06
    BETA_MIN, BETA_MAX = 0.15, 0.55

    def reset(self, cfg, fleet, spec, hubs, black, ml, recall, fp):
        self.spec, self.hubs, self.black, self.ml = spec, hubs, black, ml
        self.recall, self.fp = recall, fp
        self.vectors = spec.get("vectors", ["W"])
        self.k = len(self.vectors)
        self.n = fleet["n"]
        # ★belief state — 리셋 시 0(감염 사전지식 없음). detected로만 seed된다.
        self.belief = np.zeros(self.n)
        self.beta = self.BETA0
        self.prev_retaken = set()     # 지난 스텝 재장악(RetakeControl) 대상
        self.prev_expected = 0.0      # 지난 스텝이 예측한 '이번 스텝 기대 detected 수'

    # ---- world-model: 웜확산 전진(belief 전파) ----
    def _spread(self, b, A, beta):
        """b_pred[i] = b[i] + (1-b[i])*(1 - Π_j(1-beta·b[j]))  over A[i][j]=1.
        mpc._mpc_choose_mode의 확산모델을 belief 전파에 사용(comp 미사용, b·A만)."""
        n = self.n
        out = b.copy()
        for i in range(n):
            if b[i] >= 1.0:
                continue
            row = A[i]
            nb = b[row]
            prod = float(np.prod(1.0 - beta * nb)) if nb.size else 1.0
            out[i] = b[i] + (1.0 - b[i]) * (1.0 - prod)
        return out

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

        # ---- 관측: detector가 true comp를 detector_q로 샘플(모든 정책 공통 채널) ----
        detected = set()
        for i in comp:
            if i in unreachable:
                continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                detected.add(i)

        realistic = self.recall < 1.0 or self.fp > 0.0

        if not realistic:
            # ---- 오라클 분기: belief 루프 OFF, reach2와 동일(비교 기준) ----
            heal_set = target_set = nonclean = comp
            retaken_now = set()
            self._plan_exec_flag = False
        else:
            detected = {i for i in detected if rng.random() < self.recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < self.fp}
            evidence = detected | (fp_set if self.USE_FP_IN_BELIEF else set())

            # ---- ⑤ 피드백(먼저): 지난 예측 vs 이번 실제 detected → BETA 보정 ----
            actual = float(len(detected))
            if self.prev_expected > 0 or actual > 0:
                if actual > self.prev_expected + 1e-9:
                    self.beta = min(self.BETA_MAX, self.beta + self.BETA_LR)
                elif actual < self.prev_expected - 1e-9:
                    self.beta = max(self.BETA_MIN, self.beta - self.BETA_LR)

            # ---- ① 예측: belief 웜확산 전진 ----
            b_pred = self._spread(self.belief, A, self.beta)

            # ---- ② 관측·융합 ----
            new_b = np.empty(n)
            for i in range(n):
                if i in evidence:
                    new_b[i] = max(b_pred[i], self.DET_BELIEF)     # 탐지=강한 증거
                else:
                    new_b[i] = b_pred[i] * self.MISS_KEEP          # 미탐지 ≠ 청정
                if i in self.prev_retaken:
                    new_b[i] *= self.RETAKE_DECAY                  # 되찾음 = 신념↓
            # unreachable 노드는 관측 불가 — 예측값 그대로 유지(리셋 금지)
            self.belief = new_b

            # ---- ③ 플래닝: belief로 대상 선정(comp 미사용) ----
            # 미탐지지만 belief 높은 노드 = 예측 감염 → 선제 자가치유(비파괴, reach3 frontier의 belief 일반화)
            missed = {
                i for i in range(n)
                if i not in evidence and i not in unreachable and self.belief[i] >= self.MISS_TH
            }
            flagged = detected | fp_set
            heal_set = flagged                   # 확정 증거 → 자가 RemoveOtherSessions
            target_set = flagged                 # 청정 드론 RetakeControl(확정 증거만 파괴적 재장악)
            nonclean = flagged                   # ★missed는 재장악 풀에 유지(reach3식): 최근접 재장악 드론을 뺏기지 않음
            self._missed = missed                # 유휴 missed 드론만 아래 acts 루프에서 자가치유
            self._plan_exec_flag = True

        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0

        # ---- ④ 실행: reach2식 세그먼트별 다중 재장악 배정 ----
        # target(감염추정)을 belief 내림차순으로 정렬 → 위험 높은 감염부터 청정 드론 배정.
        if realistic and self._plan_exec_flag:
            order = {i: -self.belief[i] for i in target_set}
        else:
            order = {i: 0.0 for i in target_set}
        assign, dejam = {}, 0
        for seg in seg_groups:
            prim = sorted([c for c in seg if c in target_set], key=lambda c: order.get(c, 0.0))
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
                acts[a] = actions.make_blue_index(3, env, a, ctx)   # 자가치유(비파괴)
                continue
            if i in assign:
                acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            elif realistic and self._plan_exec_flag and i in self._missed:
                # ★유휴 예측감염(belief≥MISS_TH) 드론만 자가 RemoveOtherSessions(비파괴).
                #   재장악에 뽑힌 드론은 위 assign 분기에서 재장악 수행(우선).
                acts[a] = actions.make_blue_index(3, env, a, ctx)
            else:
                if red_jam > 0:
                    dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)   # de-jam / 감시

        # ---- ⑤ 피드백 상태 저장: 재장악 대상 + 다음 스텝 기대 detected 수 ----
        if realistic:
            self.prev_retaken = set(assign.values())
            reach_mask = np.array([1.0 if i not in unreachable else 0.0 for i in range(n)])
            self.prev_expected = float(self.recall * np.sum(b_pred * reach_mask))

        restored = min(dejam, red_jam)
        lost = len(unreachable - comp)                       # 감염∩도달불가 이중차감 방지
        avail = max(0.0, (n - len(comp) - lost - (red_jam - restored) - min(inj, max(0, n - len(comp)))) / n)
        return acts, avail
