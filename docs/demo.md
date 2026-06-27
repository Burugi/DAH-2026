# Demo & visualisation

## Example animation

`render.py` draws the fleet per step (CC3 has no built-in renderer). The GIF below
was produced with — exactly — the example command:

```bash
python src/viz/render.py sweep_sweep_c64d9f25/rule_vs_rule --gif
# -> results/sweep_sweep_c64d9f25/rule_vs_rule/figs/animation.gif
```

![rule vs rule, per-step animation](example_animation.gif)

Legend: UAV = triangle, UGV = square, red fill = compromised, blue ring = jammed,
orange arrow = GPS spoof (true → reported), yellow ring = detected. The right panel
shows the step counter, agent types, live counts and detection F1.

## Demo video

- Link: _TODO — paste the screen-recording / presentation link here._

## All visualisation commands

```bash
# interactive viewer (SPACE play/pause, <-/-> step, R reset, ESC quit)
python src/viz/render.py sweep_sweep_<hash>/rule_vs_rule

# per-step animated GIF  -> results/<matchup>/figs/animation.gif
python src/viz/render.py sweep_sweep_<hash>/rule_vs_rule --gif

# single preview frame   -> results/<matchup>/figs/pygame_preview.png
python src/viz/render.py sweep_sweep_<hash>/rule_vs_rule --smoke

# static figures         -> results/<matchup>/figs/*.png
python src/viz/plot.py sweep_sweep_<hash>/rule_vs_rule
```

`sweep.py` also writes `figs/animation.gif` for **every** matchup automatically
(disable with `--no-gif`).

## Where the figures are

Per matchup `results/sweep_*/<red>_vs_<blue>/figs/`:

| file | shows |
| --- | --- |
| `g_compromise_curve.png` | # compromised drones vs step (attack progression) |
| `a_reward.png` | cumulative blue reward vs step (defense) |
| `animation.gif` | per-step pygame animation |
| `b_compromise.png` | worm-spread heatmap |
| `c_snr_jam.png` / `d_gps_spoof.png` | jamming / GPS-spoof telemetry |
| `e_fleet_map.png` | fleet snapshots |
| `f_defense.png` | detection vs truth + GPS mitigation |

Top-level `results/sweep_*/`:

- `grid_heatmaps.png` — 3×3 of final compromise / blue reward / jam-F1
- `grid_curves.png` — compromise-vs-step for all 9 matchups
- `summary.csv` — every metric for every matchup
