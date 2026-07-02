"""공격 강도 스트레스 테스트.

starting_num_red를 1→8로 높여가며 rule / react / hier_h2의 D_mult 변화를 측정.
각 강도 × 시나리오 × 에이전트 × seed 조합을 직접 rollout하고 결과를 집계.

Usage:
    python src/run_stress.py
"""
import os, sys, csv, json, time
import numpy as np

SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

import yaml
from agents.multiagent import BLUE_MULTIAGENT_TYPES
from agents import brains
from agents.brains import RED_BRAINS, use_rl
from agents.rl import ensure_trained
import run as _run
from viz import score
from scenarios import load_scenario

# ── 실험 설정 ──────────────────────────────────────────────────────────────────
INTENSITIES = [
    {"label": "극경(1)",  "starting_num_red": 1, "red_spawn_rate": 0.05},
    {"label": "경(2)",    "starting_num_red": 2, "red_spawn_rate": 0.10},
    {"label": "중(3)",    "starting_num_red": 3, "red_spawn_rate": 0.20},
    {"label": "강(5)",    "starting_num_red": 5, "red_spawn_rate": 0.30},
    {"label": "극강(8)",  "starting_num_red": 8, "red_spawn_rate": 0.40},
]

SCENARIOS   = ["A_CONN", "A7", "A1", "A_MV"]
BLUE_TYPES  = ["rule", "react", "hier_h2", "hier_h3_tight"]
SEEDS       = list(range(10))

BASE_CFG = {
    "name": "stress",
    "fleet": {"n_uav": 12, "n_ugv": 6, "grid": 100, "max_link": 40},
    "steps": 40,
    "seeds": SEEDS,
    "red_types": ["rule"],
    "blue_types": BLUE_TYPES,
    "sim": {"max_length_data_links": 40},
    "defense": {"detector": "multisensor", "snr_thresh": 6,
                "gps_thresh": 8, "response": "safe_mode"},
    "env": "B",
}

OUT_DIR = os.path.join(ROOT, "results", "stress_test")
os.makedirs(OUT_DIR, exist_ok=True)


def run_one(cfg, seed, blue_type):
    blue_brain = BLUE_MULTIAGENT_TYPES[blue_type](cfg["fleet"]["n_uav"] + cfg["fleet"]["n_ugv"]) \
                 if blue_type in BLUE_MULTIAGENT_TYPES else None
    fleet, reward, red_owned, defence = _run.rollout(
        cfg, seed, red_type="rule", blue_type=blue_type)
    T = len(reward)
    n = fleet["n"]
    V = score.availability(red_owned, fleet["link_up"][:T])
    fc  = float(red_owned[-1].sum() / n)
    cauc = float(red_owned.sum(1).mean() / n)
    cf1 = defence["metrics"].get("comp", {}).get("F1", 0.0)
    D_core = float(np.mean([1-fc, 1-cauc, cf1]))
    return round(D_core * V, 3)


def main():
    # RL 사전 학습
    print("RL pre-train...")
    use_rl(*ensure_trained(BASE_CFG, episodes=200))

    rows = []
    t0 = time.time()

    for intens in INTENSITIES:
        for scen_id in SCENARIOS:
            cfg = dict(BASE_CFG)
            cfg["sim"] = dict(BASE_CFG["sim"])
            cfg["sim"]["starting_num_red"] = intens["starting_num_red"]
            cfg["sim"]["red_spawn_rate"]   = intens["red_spawn_rate"]
            cfg = load_scenario(scen_id, cfg)

            for blue in BLUE_TYPES:
                dms = []
                for seed in SEEDS:
                    try:
                        dm = run_one(cfg, seed, blue)
                        dms.append(dm)
                    except Exception as e:
                        dms.append(0.0)
                mean_dm = round(float(np.mean(dms)), 3)
                std_dm  = round(float(np.std(dms)),  3)
                rows.append({
                    "intensity":       intens["label"],
                    "num_red":         intens["starting_num_red"],
                    "spawn_rate":      intens["red_spawn_rate"],
                    "scenario":        scen_id,
                    "blue_type":       blue,
                    "D_mult_mean":     mean_dm,
                    "D_mult_std":      std_dm,
                })
                print(f"  [{intens['label']}] {scen_id} vs {blue:<12}: "
                      f"D_mult={mean_dm:.3f} ± {std_dm:.3f}")

    # 저장
    csv_path = os.path.join(OUT_DIR, "stress_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # 요약 테이블 출력
    print(f"\n=== 공격 강도별 D_mult (4개 시나리오 평균) ===\n")
    from collections import defaultdict
    by_intens_blue = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_intens_blue[r["intensity"]][r["blue_type"]].append(r["D_mult_mean"])

    header = f"{'강도':<10}" + "".join(f"  {b:<14}" for b in BLUE_TYPES)
    print(header)
    print("-" * (10 + 16 * len(BLUE_TYPES)))
    for intens in INTENSITIES:
        lbl = intens["label"]
        line = f"{lbl:<10}"
        for b in BLUE_TYPES:
            vals = by_intens_blue[lbl][b]
            line += f"  {np.mean(vals):.3f}          "
        print(line)

    print(f"\n총 소요: {time.time()-t0:.1f}s")
    print(f"결과 저장: {csv_path}")


if __name__ == "__main__":
    main()
