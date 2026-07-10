"""EvoBlue θ 오프라인 최적화 — 성능 상한 추정 (Phase 4).

파라미터화된 반사 정책 EvoBlue의 임계값 벡터 θ(8D)를 (μ,λ) 진화전략/랜덤서치로
학습 시나리오군에서 평균 D_mult가 최대가 되도록 최적화한다. 두 예산 지점에서 각각:
  · k=inf : graph(0.794)를 넘는가? → "무제한이면 아키텍처보다 파라미터가 병목"
  · k=6   : whittle(0.746)을 넘는가? → "예산 제약 하에서도 튜닝된 반사가 충분한가"

과적합 방지: train 시나리오(홀수 인덱스)로 최적화 → 최종은 전체 23시나리오로 평가하고
train/heldout 점수를 함께 보고. θ는 results/evo/theta_<tag>.json에 저장.

Usage:
    python src/run_evo.py --budget inf --gens 25 --pop 16
    python src/run_evo.py --budget 6   --gens 25 --pop 16
"""
import os, sys, json, time, argparse
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import run as _run
from viz import score
from scenarios import load_scenario
from agents.brains import use_rl
from agents.rl import ensure_trained
from agents.experimental import EvoBlue
from run_arch_matrix import SCENARIOS, BASE_CFG

OUT = os.path.join(ROOT, "results", "evo")
os.makedirs(OUT, exist_ok=True)

# 학습/홀드아웃 분할 (과적합 점검용)
TRAIN = [s for i, s in enumerate(SCENARIOS) if i % 2 == 0]
HELDOUT = [s for i, s in enumerate(SCENARIOS) if i % 2 == 1]


def eval_theta(theta, scens, seeds, reds, budget, cfg_cache):
    """θ 정책의 평균 D_mult (scens × seeds × reds)."""
    vals = []
    for scen in scens:
        cfg = cfg_cache[scen]
        for seed in seeds:
            for red in reds:
                n = cfg["fleet"]["n_uav"] + cfg["fleet"]["n_ugv"]
                fleet, reward, red_owned, df = _run.rollout(
                    cfg, seed, red_type=red, blue_type="__evo__",
                    blue_brain=EvoBlue(n, theta))
                vals.append(score.d_mult_single(red_owned, fleet["link_up"][:len(red_owned)]))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="inf")
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=4, help="학습 평가 시드 수(속도)")
    ap.add_argument("--reds", default="rule,llm,rl")
    a = ap.parse_args()

    budget = a.budget if a.budget == "inf" else int(a.budget)
    tag = "kinf" if a.budget == "inf" else f"k{a.budget}"
    seeds = list(range(a.seeds)); reds = a.reds.split(",")
    base = dict(BASE_CFG, seeds=seeds, red_types=reds, blue_types=["evo"])
    use_rl(*ensure_trained(base, 100))

    # cfg 캐시(예산 주입)
    cfg_cache = {s: load_scenario(s, dict(base, sim=dict(base["sim"]),
                                          recovery_budget=budget)) for s in SCENARIOS}

    rng = np.random.default_rng(0)
    dim = EvoBlue.DIM
    # (μ,λ) ES: 평균 벡터 + 등방성 시그마
    mean = np.array(EvoBlue.DEFAULT, float)
    sigma = 0.25
    best = (list(mean), -1.0)
    t0 = time.time()
    print(f"EvoBlue 최적화 [budget={a.budget}]  train={len(TRAIN)}scen × {a.seeds}seed × {len(reds)}red")
    for g in range(a.gens):
        pop = np.clip(mean + sigma * rng.standard_normal((a.pop, dim)), 0.0, 1.0)
        fits = np.array([eval_theta(list(ind), TRAIN, seeds, reds, budget, cfg_cache)
                         for ind in pop])
        order = np.argsort(-fits)
        elite = pop[order[:max(2, a.pop//4)]]
        mean = elite.mean(0)                              # 엘리트 평균으로 이동
        sigma = max(0.05, sigma * 0.92)                   # 점진 수렴
        if fits[order[0]] > best[1]:
            best = (list(pop[order[0]]), float(fits[order[0]]))
        print(f"  gen {g:2d}  best_train={best[1]:.4f}  gen_best={fits[order[0]]:.4f}  "
              f"sigma={sigma:.3f}  {time.time()-t0:.0f}s")

    theta = best[0]
    # 최종 평가: train / heldout / 전체 (더 많은 시드로 확인)
    ev_seeds = list(range(8))
    tr = eval_theta(theta, TRAIN, ev_seeds, reds, budget, cfg_cache)
    ho = eval_theta(theta, HELDOUT, ev_seeds, reds, budget, cfg_cache)
    al = eval_theta(theta, SCENARIOS, ev_seeds, reds, budget, cfg_cache)
    json.dump({"theta": theta, "budget": a.budget, "train_dmult": tr,
               "heldout_dmult": ho, "all_dmult": al},
              open(os.path.join(OUT, f"theta_{tag}.json"), "w"), indent=1)
    print(f"\n[budget={a.budget}] 최적 θ 저장 → results/evo/theta_{tag}.json")
    print(f"  train={tr:.4f}  heldout={ho:.4f}  전체(8seed)={al:.4f}")
    ref = {"inf": "graph=0.794", "6": "whittle=0.746"}.get(a.budget, "")
    print(f"  기준선: {ref}  → {'초과' if al > (0.794 if a.budget=='inf' else 0.746) else '미달'}")


if __name__ == "__main__":
    main()
