"""Run the full attack(3) x defense(3) comparison and visualise the 3x3 grid.

    python sweep.py configs/sweep.yaml
    python sweep.py configs/sweep.yaml --seeds 1 --steps 20 --episodes 20 --fresh   # smoke

For every (red_type, blue_type) in {rule, llm, rl}^2 this runs the matchup, saves
its own results/<sweep>/<red>_vs_<blue>/ (log.csv, arrays.npz, meta.json, figs/),
then writes a top-level summary.csv plus 3x3 heatmaps and a 9-panel curve grid.
"""
import os, sys, json, csv, time, hashlib, argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")     # headless pygame for GIF export
import run
from viz import plot, render
from agents.brains import use_rl
from agents.rl import ensure_trained

SUMMARY_COLS = ["red_type", "blue_type", "final_compromise", "peak_compromise",
                "time_to_first_compromise", "compromise_auc", "blue_reward_total",
                "recovered", "comp_F1", "jam_F1", "gps_F1", "gps_err_before", "gps_err_after"]


def _heatmap(ax, M, reds, blues, title, fmt="{:.2f}", cmap="viridis"):
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(blues))); ax.set_xticklabels(blues)
    ax.set_yticks(range(len(reds))); ax.set_yticklabels(reds)
    ax.set_xlabel("blue (defense)"); ax.set_ylabel("red (attack)"); ax.set_title(title)
    for i in range(len(reds)):
        for j in range(len(blues)):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def grid_figures(sweep_dir, reds, blues, metrics, curves, name):
    """3x3 heatmaps + 9-panel compromise-vs-step grid."""
    M = lambda key: np.array([[metrics[(r, b)][key] for b in blues] for r in reds])
    panels = [("final_compromise", "final compromised fraction", "{:.2f}", "Reds"),
              ("blue_reward_total", "blue cumulative reward", "{:.0f}", "viridis"),
              ("comp_F1", "worm/compromise detection F1", "{:.2f}", "Greens")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, ttl, fmt, cmap) in zip(axs, panels):
        _heatmap(ax, M(key), reds, blues, ttl, fmt, cmap)
    fig.suptitle(f"{name} — 3x3 attack x defense (mean over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, "grid_heatmaps.png"), dpi=115, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(len(reds), len(blues), figsize=(4 * len(blues), 3 * len(reds)),
                            sharex=True, sharey=True, squeeze=False)
    for i, r in enumerate(reds):
        for j, b in enumerate(blues):
            ax = axs[i][j]
            ax.plot(curves[(r, b)], ".-", color="firebrick")
            ax.set_title(f"red={r} vs blue={b}", fontsize=9)
            ax.grid(alpha=.3)
            if i == len(reds) - 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel("# compromised")
    fig.suptitle(f"{name} — compromised drones over time (3x3)")
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, "grid_curves.png"), dpi=115, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=os.path.join(SRC, "configs", "sweep.yaml"))
    ap.add_argument("--seeds", type=int, help="override: use seeds 0..N-1")
    ap.add_argument("--steps", type=int, help="override episode length")
    ap.add_argument("--episodes", type=int, default=200, help="rl training episodes")
    ap.add_argument("--fresh", action="store_true", help="retrain rl even if cached")
    ap.add_argument("--no-gif", action="store_true", help="skip per-matchup pygame GIF")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.seeds:
        cfg["seeds"] = list(range(a.seeds))
    if a.steps:
        cfg["steps"] = a.steps
    reds = cfg.get("red_types", ["rule", "llm", "rl"])
    blues = cfg.get("blue_types", ["rule", "llm", "rl"])

    if "rl" in reds or "rl" in blues:
        use_rl(*ensure_trained(cfg, a.episodes, fresh=a.fresh))

    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    sweep_dir = os.path.join(RESULTS, f"sweep_{cfg['name']}_{h}")
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"sweep -> {os.path.relpath(sweep_dir, ROOT)}  "
          f"red={reds} blue={blues} seeds={cfg['seeds']} steps={cfg['steps']}")

    metrics, curves, rows, t0 = {}, {}, [], time.time()
    for r in reds:
        for b in blues:
            out = os.path.join(sweep_dir, f"{r}_vs_{b}")
            res = {seed: run.rollout(cfg, seed, r, b) for seed in cfg["seeds"]}
            m = run.save_run(cfg, out, r, b, res)
            plot.make_figs(out)
            if not a.no_gif:
                render.save_gif(out)
            ros = [ro.sum(1) for (_, _, ro, _) in res.values()]
            tmin = min(len(x) for x in ros)
            curves[(r, b)] = np.mean([x[:tmin] for x in ros], axis=0)
            metrics[(r, b)] = m
            rows.append([r, b] + [m[k] for k in SUMMARY_COLS[2:]])
            print(f"  {r:5} vs {b:5}: final_comp={m['final_compromise']:.2f} "
                  f"blueR={m['blue_reward_total']:.0f} compF1={m['comp_F1']} "
                  f"jamF1={m['jam_F1']} gpsF1={m['gps_F1']}")

    with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([SUMMARY_COLS] + rows)
    grid_figures(sweep_dir, reds, blues, metrics, curves, cfg["name"])

    hdr = "red\\blue"
    print(f"\n{hdr:12}" + "".join(f"{b:>14}" for b in blues) + "   (final_comp | blueR)")
    for r in reds:
        line = f"{r:12}"
        for b in blues:
            m = metrics[(r, b)]
            line += f"{m['final_compromise']:.2f}|{m['blue_reward_total']:>6.0f}".rjust(14)
        print(line)
    print(f"\n-> {os.path.relpath(sweep_dir, ROOT)}/  (summary.csv, grid_heatmaps.png, "
          f"grid_curves.png)  {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
