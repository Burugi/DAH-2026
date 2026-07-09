# -*- coding: utf-8 -*-
"""방어 정책 공통 베이스 + 토폴로지/재장악 헬퍼.

방어 모델(reach2·jy_hvt·rag_guided 등)이 공통으로 쓰는 최소 인프라를 한 곳에 모음.
(개인 벤치 하네스에서 팀 레포로 추출 — 범용 인프라, 경쟁 우위 아님. 채점 러너는 미포함.)

Policy 인터페이스: reset(cfg,fleet,spec,hubs,black,ml,recall,fp) + step(comp,pos,env,live,ip2d,rng)->(acts,avail)
"""
import numpy as np
from agents import actions        # 팀 blue 행동 디스패치(make_blue_index·action_index_map)

JAM_VECS = {"J", "B"}                                   # 재밍 계열 공격 레인
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}     # 공격 레인 → 행동 인덱스


def adjacency(pos, ml):
    """드론 위치 → max_link(ml) 내 연결 인접행렬(자기 제외)."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return (d < ml) & (d > 0)


def components(present, A, n):
    """present 노드들의 연결성분 리스트(인접행렬 A 기준)."""
    seen, out = set(), []
    for s in present:
        if s in seen:
            continue
        cc, st = set(), [s]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u)
            cc.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen:
                    st.append(v)
        out.append(cc)
    return out


def retake_target(env, a, node, ip2d, sleep):
    """블루 에이전트 a의 RetakeControl 액션 중 대상 node를 겨누는 인덱스."""
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node:
            return i
    c = idx.get("RetakeControl", [])
    return c[0][0] if c else sleep


class DefensePolicy:
    """방어 두뇌 공통 어댑터. rollout이 매 스텝 step()을 호출하고 (행동 dict, 가용성)을 받음."""
    needs_pretrain = False
    name = "base"

    def reset(self, cfg, fleet, spec, hubs, black, ml, recall, fp):
        """에피소드 시작 시 1회. 토폴로지·시나리오 파라미터 보관."""

    def step(self, comp, pos, env, live, ip2d, rng):
        """한 스텝 결정 → (acts {agent: wrapper_index}, avail 0~1).
        comp=실제 감염(채점 기준; 결정엔 탐지신념 사용 권장), pos=(n,2) 좌표, ip2d=ip→drone id."""
        raise NotImplementedError

    def load(self, path):
        """사전학습 가중치 로드(torch/tabular 계열만)."""

    def train(self, cfg):
        """온라인/사전 학습(선택)."""
