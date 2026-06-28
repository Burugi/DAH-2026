"""Bulk synthetic telemetry dataset for detector training (no sim, so it scales).

    python make_dataset.py configs/sweep.yaml --seeds 50 --out data/synth.csv

One CSV row per (seed, step, entity) with telemetry and jam/gps ground-truth labels.
Use it to train/evaluate anomaly detectors offline, or to augment with more seeds.
"""
import os, sys, csv, argparse
import numpy as np
import yaml

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from sim.fleet import generate_fleet

COLS = ["seed", "step", "entity_id", "entity_type", "pos_true_x", "pos_true_y",
        "pos_rep_x", "pos_rep_y", "gps_err", "snr", "link_up", "label_jam", "label_gps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, default=50, help="number of seeds (0..N-1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    fl = cfg["fleet"]
    out = args.out or os.path.join(ROOT, "data", f"{cfg['name']}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    rows = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS)
        for seed in range(args.seeds):
            fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], seed,
                                   fl["grid"], cfg.get("attacks"), fl.get("max_link", 40))
            types = fleet["types"]
            for t in range(fleet["steps"]):
                for e in range(fleet["n"]):
                    w.writerow([seed, t, e, types[e],
                                *np.round(fleet["pos_true"][t, e], 2), *np.round(fleet["pos_rep"][t, e], 2),
                                round(float(fleet["gps_err"][t, e]), 2), round(float(fleet["snr"][t, e]), 2),
                                int(fleet["link_up"][t, e]), int(fleet["label_jam"][t, e]),
                                int(fleet["label_gps"][t, e])])
                    rows += 1
    pos = sum(1 for _ in open(out, encoding="utf-8")) - 1
    print(f"{args.seeds} seeds -> {out} ({pos} rows)")


if __name__ == "__main__":
    main()
