# -*- coding: utf-8 -*-
"""MPC(모델 예측 제어) 방어 두뇌 — `exp_advanced.py`의 mpc를 23시나리오 하네스에 이식.

결심 전 각 MODE{retake, block, allow}에 대해 내부 웜 확산을 k=3스텝 전진 시뮬레이션하고
예측 (1−점령)×가용성을 최대화하는 모드를 고른다. 코디네이터-패밀리로 감싸(토폴로지·탐지·오버레이
동일 처리) 코디네이터/통합방어와 같은 축에서 비교 가능하게 했다.
"""
import numpy as np

from harness import JAM_VECS, actions, adjacency, components, DefensePolicy

# (spread_scale, retake_rate, block_cost) — 스텝당
_MODE_PARAM = {"retake": (1.0, 0.5, 0.05), "block": (0.4, 0.2, 0.40), "allow": (1.0, 0.0, 0.00)}


def _mpc_choose_mode(seed_comp, A, n, beta=0.35, k=3):
    """탐지된 감염을 seed로 각 모드의 k스텝 확산을 예측 → (1−점령)×가용성 최대 모드."""
    best_mode, best_val = "retake", -1.0
    for mode, (ss, rt, bc) in _MODE_PARAM.items():
        p = np.zeros(n)
        for i in seed_comp:
            p[i] = 1.0
        be = beta * ss
        for _ in range(k):
            newp = p.copy()
            for j in range(n):
                if p[j] < 1.0:
                    prod = float(np.prod(1 - be * p[A[j]])) if A[j].any() else 1.0
                    newp[j] = p[j] + (1 - p[j]) * (1 - prod)
            p = newp * (1 - rt)
        comp_pred = float(p.mean())
        avail_pred = max(0.0, 1 - comp_pred - bc)
        val = (1 - comp_pred) * avail_pred
        if val > best_val:
            best_val, best_mode = val, mode
    return best_mode


def _mode_aid(mode, in_heal, has_det, fr, jam):
    if in_heal:
        return 3                                        # 자가복구
    if mode == "block":
        return 6 if fr else (4 if has_det else 1)
    if mode == "retake":
        return 4 if has_det else ("dejam" if jam else 1)
    return 4 if has_det else ("dejam" if jam else 7)    # allow


class MPCPolicy(DefensePolicy):
    name = "mpc"

    def _choose_mode(self, detected, comp, A, n, jam):
        """모드 선택 훅 — MPC는 내부 시뮬 argmax. (LLM 지휘관은 이걸 오버라이드)"""
        return _mpc_choose_mode(detected, A, n) if detected else "retake"

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
        unreachable = hubs | black | (present - big)    # 코디네이터-패밀리(relay 없음)

        detected = set()
        for i in comp:
            if i in unreachable:
                continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                detected.add(i)
        if self.recall < 1.0 or self.fp > 0.0:          # 현실 탐지
            detected = {i for i in detected if rng.random() < self.recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < self.fp}
            heal_set = detected | fp_set
        else:
            heal_set = comp
        has_det = len(detected) > 0
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        jam = red_jam > 0
        inj = len(comp) if spec.get("inject") else 0
        mode = self._choose_mode(detected, comp, A, n, jam)      # 서브클래스가 오버라이드 가능

        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        acts, blocks, dejam = {}, 0, 0
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable:
                acts[a] = actions.make_blue_index(0, env, a, ctx)
                continue
            fr = any(d != i and np.linalg.norm(pos[i] - pos[d]) < self.ml for d in detected)
            aid = _mode_aid(mode, i in heal_set, has_det, fr, jam)
            if aid == "dejam":
                dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)
            else:
                if aid == 6:
                    blocks += 1
                acts[a] = actions.make_blue_index(aid, env, a, ctx)

        restored = min(dejam, red_jam)
        lost = len(unreachable - comp)                       # 감염∩도달불가 이중차감 방지
        avail = max(0.0, (n - len(comp) - lost - blocks - (red_jam - restored) - min(inj, max(0, n - len(comp)))) / n)
        return acts, avail
