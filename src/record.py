# -*- coding: utf-8 -*-
"""방어 모델(DefensePolicy) 실행을 대시보드 번들로 기록 — 데모/영상용.

score.py는 점수만 출력한다. 이 러너는 같은 rollout을 results/<run_id>/
(log.csv · arrays.npz · meta.json)로 저장해 기존 시각화 도구를 그대로 쓴다.

    python src/record.py --scenario A1                     # 기본 모델 = rag-guided(HVT+RAG)
    python src/record.py --scenario A7 --recall 0.75 --fp 0.1
    python src/viz/dashboard.py results/<run_id> --png     # 대시보드 HTML + 프리뷰
    python src/viz/render.py results/<run_id> --gif        # 함대 애니메이션 GIF

(구 run_hvt.py를 모델 선택형으로 일반화 — 모델 로딩은 score.load_policy 재사용.)
"""
import os, sys, time, argparse
import numpy as np
import yaml

SRC = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(SRC), "results")
sys.path.insert(0, SRC)

import run
from agents import actions, brains
from agents.actions import RED_N, BLUE_DECISION_N
from agents.defense_base import adjacency
from sim.defense import run_defense
from sim.fleet import truncate
from viz import score as vscore
from score import load_policy, get_spec, make_red, BASE_CFG

_NAME2BLUE = {"RemoveOtherSessions": 3, "RetakeControl": 4,
              "BlockTraffic": 6, "AllowTraffic": 7}


def blue_action_tape(env, acts):
    """이 스텝 blue 행동 -> (대표 행동 id, 카탈로그별 카운트). 대시보드 전술 로그용."""
    row = np.zeros(BLUE_DECISION_N, np.int16)
    for a, widx in acts.items():
        action = env.agent_actions.get(a, {}).get(widx)
        name = type(action).__name__ if action is not None else "Sleep"
        row[_NAME2BLUE.get(name, 1)] += 1
    rep = int(row.argmax()) if row.sum() else 1
    return rep, row


def rollout(cfg, spec, policy, seed, recall, fp):
    """한 에피소드 실행 + 기록. (fleet, reward, red_owned, defence) 반환."""
    vectors = spec.get("vectors", ["W"]); tempo = spec.get("tempo", 1.0)
    ml = cfg["fleet"].get("max_link", 40)
    cfg2 = cfg
    if spec.get("start_red"):
        cfg2 = dict(cfg); cfg2["sim"] = dict(cfg["sim"])
        cfg2["sim"]["starting_num_red"] = spec["start_red"]
    fleet, cyborg, env, ip2d = run.build_env(cfg2, seed, make_red(vectors, tempo))
    n = fleet["n"]; rng = np.random.default_rng(seed + 4)

    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:spec.get("frag_K", 0)]) if spec.get("frag_K") else set()
    kbl = int(round(spec.get("blackout_p", 0.0) * n))
    black = set(int(x) for x in rng.choice(n, size=kbl, replace=False)) if kbl else set()
    policy.reset(cfg, fleet, spec, hubs, black, ml, recall, fp)

    reward = np.zeros(cfg["steps"])
    red_owned = np.zeros((cfg["steps"], n), np.int8)
    red_log, blue_log = [], []
    brains.pop_red_actlog()

    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        live = [a for a in env.active_agents if a in env.agent_actions]
        acts, _ = policy.step(comp, pos, env, live, ip2d, rng)
        blue_log.append(blue_action_tape(env, acts))
        _, rew, done, _ = env.step(acts)
        red_log.append(run._summ(brains.pop_red_actlog(), RED_N))
        reward[t] = float(np.mean(list(rew.values()))) if rew else 0.0
        for d in run.compromised_drones(cyborg, n):
            red_owned[t, d] = 1
        if all(done.values()):
            reward, red_owned = reward[:t + 1], red_owned[:t + 1]
            truncate(fleet, t + 1)
            break

    T = len(reward)
    a_t, d_t = vscore.per_step(red_owned, fleet["link_up"][:T])
    defence = run_defense(cfg, fleet, seed, red_owned)
    defence.update(
        red_act=np.array([r[0] for r in red_log], np.int16),
        blue_act=np.array([b[0] for b in blue_log], np.int16),
        red_cnt=np.stack([r[1] for r in red_log]) if red_log else np.zeros((T, RED_N), np.int16),
        blue_cnt=np.stack([b[1] for b in blue_log]) if blue_log else np.zeros((T, BLUE_DECISION_N), np.int16),
        a_t=a_t, d_t=d_t)
    return fleet, reward, red_owned, defence


def main():
    ap = argparse.ArgumentParser(description="방어 모델 실행 기록 (대시보드 번들 생성)")
    ap.add_argument("--model", default="rag-guided", help="rag-guided(기본, HVT+RAG) · hvt · reach2")
    ap.add_argument("--scenario", default="A17", help="시나리오 id (A1~A21, A-CONN, A-MV)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002])
    ap.add_argument("--recall", type=float, default=1.0, help="현실 탐지율. 실전 0.75")
    ap.add_argument("--fp", type=float, default=0.0, help="현실 오탐율. 실전 0.1")
    a = ap.parse_args()

    spec, name = get_spec(a.scenario)
    cfg = dict(BASE_CFG); cfg["name"] = f"{a.scenario} {name}".strip()
    red_label = "+".join(spec.get("vectors", ["W"]))
    tag = f"_r{a.recall}_fp{a.fp}" if (a.recall < 1.0 or a.fp > 0.0) else ""
    run_id = f"{a.model.replace('-', '_')}_{a.scenario}{tag}"
    out = os.path.join(RESULTS, run_id)

    print(f"{a.model}  {cfg['name']}  vectors={red_label} recall={a.recall} fp={a.fp}")
    t0, results = time.time(), {}
    for seed in a.seeds:
        fleet, reward, red_owned, df = rollout(cfg, spec, load_policy(a.model, a.scenario),
                                               seed, a.recall, a.fp)
        results[seed] = (fleet, reward, red_owned, df)
        print(f"  seed {seed}: compromised={int(red_owned[-1].sum())}/{fleet['n']} "
              f"peak={int(red_owned.sum(1).max())}")

    metrics = run.save_run(cfg, out, red_label, a.model, results)
    print(f"-> results/{run_id}/  ({round(time.time() - t0, 1)}s)  "
          f"final_comp={metrics['final_compromise']} availability={metrics['availability']} "
          f"defense={metrics['defense_score']}")
    print(f"   dashboard: python src/viz/dashboard.py results/{run_id} --png")
    print(f"   animation: python src/viz/render.py results/{run_id} --gif")


if __name__ == "__main__":
    main()
