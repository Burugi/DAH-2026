# jy_reach3 — NeuroGuard 방어모델 구조 (원본: neuroguard/agents/reach3.py)
# HybridReach — frontier 선제 자가치유  · 실동작 통합은 본선 구조에서.
# -*- coding: utf-8 -*-
"""도달성 특화 방어 v3 — frontier 선제 자가치유로 '놓친 감염'을 catch.

ReachV2(reach2) 대비 개선점(현실적 탐지 recall<1·fp>0 강건성 목표):
  * reach2의 다중재장악(REDUNDANCY=3)·무제한 relay 재연결·de-jam 결합·
    가용성 회계(`lost=len(unreachable-comp)`)는 **전부 그대로 유지**.
  * ★변경점 — heal 대상(heal_set)을 `detected`(+fp)에 더해
    **detected의 청정 이웃(frontier)** 까지 확장한다.
      frontier = {j : j가 청정으로 보이고(=nonclean/unreachable 아님),
                      A[i][j]로 detected인 i에 인접, 그런 detected 이웃 수 ≥ FRONTIER_THRESH}.
    현실적 탐지가 recall<1이라 웜이 이웃으로 퍼졌는데 그 이웃을 놓치는 경우가 생긴다.
    frontier 드론은 **자가 RemoveOtherSessions(aid 3)** 를 수행한다 — 실제 감염이면
    레드 세션을 제거해 놓친 감염을 잡고, 청정이면 무해(제거할 게 없음).
  * ★왜 재장악(RetakeControl=Restore)이 아니라 heal(RemoveOtherSessions)인가:
    RetakeControl은 호스트를 Restore해 청정 노드에도 파괴적(세션 손실·재노출)이라
    frontier의 다수를 차지하는 '실제 청정' 노드를 도리어 악화시킨다(실측 0.900→0.86).
    RemoveOtherSessions는 비파괴적이라 놓친 감염만 선별적으로 청소한다.
    (frontier 드론이 원래 하던 기본행동은 Sleep(aid 1)이라 잃는 게 없다.)
  * **오라클 누수 없음** — 현실 분기에서 heal/재장악 대상은 오직 detected/fp/frontier.
    `comp`(true 감염)는 현실 분기 결정에 **일절 쓰지 않는다**(frontier도 A·detected로만).
    오라클 분기(recall=1)는 reach2와 완전히 동일.

FRONTIER_MODE:
  "heal"   frontier 드론이 자가 RemoveOtherSessions(비파괴, 기본·권장)
  "retake" frontier를 청정 드론이 RetakeControl(파괴적, 비교용)
  "off"    frontier 비활성(=reach2 동치)
"""
import numpy as np

from neuroguard.agents.base import DefensePolicy
from neuroguard.harness import JAM_VECS, actions, adjacency, components, retake_target


class HybridReach(DefensePolicy):
    name = "reach3"
    R_relay = None            # None = 무제한 재연결(reach2와 동일)
    REDUNDANCY = 3            # detected 1대당 청정 재장악 드론 수(reach2 최적=3)
    FRONTIER_MODE = "heal"   # heal(권장) | retake | off
    FRONTIER_THRESH = 1      # frontier 편입 조건: detected 이웃 수 ≥ THRESH
    FRONTIER_RED = 1         # retake 모드에서 frontier 1대당 청정 재장악 드론 수

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
        reconnectable = list(isolated) + list(black)
        cap = len(reconnectable) if self.R_relay is None else self.R_relay
        reconnected = set(reconnectable[:cap])
        unreachable = (hubs | black | isolated) - reconnected
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

        frontier_set = set()
        realistic = self.recall < 1.0 or self.fp > 0.0
        if realistic:
            detected = {i for i in detected if rng.random() < self.recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < self.fp}
            heal_set = target_set = nonclean = detected | fp_set
            if self.FRONTIER_MODE != "off":
                # ★frontier: detected의 청정 이웃(true comp 미사용 — A와 detected로만).
                deg = {}
                for i in detected:
                    row = A[i]
                    for j in range(n):
                        if row[j] and j not in nonclean and j not in unreachable:
                            deg[j] = deg.get(j, 0) + 1
                frontier_set = {j for j, c in deg.items() if c >= self.FRONTIER_THRESH}
                # heal 모드: frontier 드론은 heal_set/nonclean에 넣지 않는다(★핵심).
                # → 재장악 드론 풀(scl)에 그대로 남겨 detected 재장악에 우선 투입하고,
                #   재장악에 안 뽑힌 '유휴' frontier 드론만 아래 acts 루프에서 자가치유한다.
                #   (fragment/blackout 시 최근접 재장악 드론을 뺏기지 않아 A4·A10 손실 방지)
        else:
            heal_set, target_set, nonclean = comp, detected, comp

        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0

        assign, dejam = {}, 0
        retake_frontier = frontier_set if self.FRONTIER_MODE == "retake" else set()
        for seg in seg_groups:
            prim = [c for c in seg if c in target_set]
            front = [c for c in seg if c in retake_frontier]
            scl = [d for d in seg if d not in nonclean]
            used = set()

            def _assign(tgts, red):
                for _ in range(red):
                    for c in tgts:
                        cand = sorted(
                            [d for d in scl if d not in used and d != c],
                            key=lambda d: np.linalg.norm(pos[d] - pos[c]),
                        )
                        if cand:
                            assign[cand[0]] = c
                            used.add(cand[0])

            _assign(prim, self.REDUNDANCY)      # detected 우선(REDUNDANCY=3)
            _assign(front, self.FRONTIER_RED)   # frontier 선제(저중복), 남는 청정 드론만

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
            elif self.FRONTIER_MODE == "heal" and i in frontier_set:
                # ★유휴 frontier 드론만 자가 RemoveOtherSessions(비파괴, 놓친 감염 선별 청소).
                #   재장악에 뽑힌 frontier 드론은 위 assign 분기에서 재장악을 수행(우선).
                acts[a] = actions.make_blue_index(3, env, a, ctx)
            else:
                if red_jam > 0:
                    dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)

        restored = min(dejam, red_jam)
        lost = len(unreachable - comp)
        avail = max(0.0, (n - len(comp) - lost - (red_jam - restored) - min(inj, max(0, n - len(comp)))) / n)
        return acts, avail
