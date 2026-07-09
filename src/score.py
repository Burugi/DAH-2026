# -*- coding: utf-8 -*-
"""방어 모델(DefensePolicy) 채점 러너 — 팀 src 자립 실행.

우리 방어 모델(agents/hvt·reach2)은 DefensePolicy 인터페이스(reset/step)라
run.py(rule/llm/rl 전용)로는 못 돌린다. 이 러너가 그 rollout+점수를 담당.

사용:
  python score.py --scenario A17 --recall 0.75 --fp 0.1 --seeds 5   # 기본 모델 = HVT+RAG
  python score.py --scenario A17 --log steps.csv                    # step별 상태 CSV

모델: rag-guided(기본, HVT+RAG — appendix/rag_guided.py, attack_class는 RAG-A
      오프라인 산출물 scenario_attack_class.json에서 주입) · hvt · reach2
채점: 채널① 방어점수 = mean(1-최종점령, 1-평균점령). ↑ 높을수록 우수.
실전조건: --recall 0.75 --fp 0.1 (미탐/오탐). 생략 시 오라클(1.0/0.0).
"""
import os, sys, argparse, csv
import numpy as np
import yaml

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

import run                                              # build_env, compromised_drones (CybORG)
from agents import brains                               # _Red (make_red 의존)
from agents.defense_base import adjacency, VEC_AIDS     # 토폴로지 헬퍼 + 공격레인→행동


def load_policy(name, scenario=None):
    """이름 → DefensePolicy 인스턴스 (매 seed 새로 생성해 상태 격리)."""
    if name == "hvt":
        from agents.hvt import HVTDefense; return HVTDefense()
    if name == "reach2":
        from agents.reach2 import ReachV2; return ReachV2()
    if name in ("rag-guided", "rag_guided"):
        # HVT+RAG: RAG-A(appendix/attack_rag)가 산출한 시나리오별 attack_class를 주입해
        # 대응 자세(봉쇄우선/복원력)를 라우팅. 임베딩 env 불필요(오프라인 산출물 사용).
        import json
        appendix = os.path.join(os.path.dirname(_SRC), "appendix")
        sys.path.insert(0, appendix)
        from rag_guided import RAGGuidedPolicy
        cls_path = os.path.join(appendix, "attack_rag", "rag_data", "scenario_attack_class.json")
        classes = json.load(open(cls_path, encoding="utf-8"))
        key = (scenario or "").replace("-", "_")
        return RAGGuidedPolicy(attack_class=classes.get(key, "unknown"))
    raise SystemExit(f"unknown model {name!r} (available: hvt, reach2, rag-guided)")


def get_spec(sid):
    """attack_scenarios.yaml에서 시나리오 sim-spec(vectors/frag_K/...) 조회."""
    path = os.path.join(_SRC, "configs", "attack_scenarios.yaml")
    raw = yaml.safe_load(open(path, encoding="utf-8"))["scenarios"]
    for s in raw:
        if s["id"] == sid:
            return {k: v for k, v in s.items() if k not in ("id", "name", "class")}, s["name"]
    raise SystemExit(f"scenario {sid!r} not found (available: {[s['id'] for s in raw]})")


def make_red(vectors, tempo):
    """시나리오 벡터(W/J/B)·tempo로 레드 브레인 생성 (harness.make_red와 동일)."""
    class _ScenarioRed(brains._Red):
        VECS = list(vectors)

        def get_action(self, obs, asp):
            if self.np_random.uniform() > tempo:
                return self._emit(0, obs)
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            o = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[o % len(self.VECS)]])), obs)
    return _ScenarioRed


BASE_CFG = yaml.safe_load(open(os.path.join(_SRC, "configs", "sweep.yaml"), encoding="utf-8"))


def episode(seed, spec, policy, recall, fp, log_rows=None, sid=""):
    """단일 에피소드 rollout → (최종점령, 평균점령, 평균가용성). log_rows!=None이면 step별 상태 적재."""
    cfg = BASE_CFG
    vectors = spec.get("vectors", ["W"]); tempo = spec.get("tempo", 1.0)
    cfg2 = cfg
    if spec.get("start_red"):
        cfg2 = dict(cfg); cfg2["sim"] = dict(cfg["sim"]); cfg2["sim"]["starting_num_red"] = spec["start_red"]
    fleet, cyborg, env, ip2d = run.build_env(cfg2, seed, make_red(vectors, tempo))
    n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40); rng = np.random.default_rng(seed + 4)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:spec.get("frag_K", 0)]) if spec.get("frag_K") else set()
    kbl = int(round(spec.get("blackout_p", 0.0) * n))
    black = set(int(x) for x in rng.choice(n, size=kbl, replace=False)) if kbl else set()

    policy.reset(cfg, fleet, spec, hubs, black, ml, recall, fp)
    cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts, avail = policy.step(comp, pos, env, live, ip2d, rng)
        af.append(max(0.0, avail))
        if log_rows is not None:
            log_rows.append([sid, seed, t, round(len(comp) / n, 3), round(max(0.0, avail), 3), len(comp)])
        _, _, done, _ = env.step(acts)
        if all(done.values()):
            break
    return cf[-1], float(np.mean(cf)), float(np.mean(af))


def main():
    ap = argparse.ArgumentParser(description="방어 모델 채점 (DefensePolicy)")
    ap.add_argument("--model", default="rag-guided",
                    help="rag-guided(기본, HVT+RAG) · hvt · reach2")
    ap.add_argument("--scenario", default="A17", help="시나리오 id (A1~A21, A-CONN, A-MV)")
    ap.add_argument("--recall", type=float, default=1.0, help="현실 탐지율(미탐). 실전 0.75")
    ap.add_argument("--fp", type=float, default=0.0, help="현실 오탐율. 실전 0.1")
    ap.add_argument("--seeds", type=int, default=5, help="시드 개수(3000~)")
    ap.add_argument("--log", default=None, help="step별 상태 CSV 저장 경로")
    a = ap.parse_args()

    spec, name = get_spec(a.scenario)
    seeds = [3000 + i for i in range(a.seeds)]
    log_rows = [] if a.log else None
    scores = []
    for s in seeds:
        final, auc, avail = episode(s, spec, load_policy(a.model, a.scenario), a.recall, a.fp, log_rows, a.scenario)
        scores.append(float(np.mean([1 - final, 1 - auc])))

    tag = "" if (a.recall == 1.0 and a.fp == 0.0) else f" (현실탐지 r{a.recall}/fp{a.fp})"
    print(f"model={a.model}  scenario={a.scenario} {name}  seeds={a.seeds}{tag}")
    print(f"방어 점수 = {np.mean(scores):.3f}   (0~1, ↑ 높을수록 방어 우수)")
    if a.log:
        with open(a.log, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "seed", "step", "compromise_frac", "availability", "n_compromised"])
            w.writerows(log_rows)
        print(f"step별 상태 → {a.log}  ({len(log_rows)} 행)")


if __name__ == "__main__":
    main()
