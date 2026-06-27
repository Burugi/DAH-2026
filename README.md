# DroneSwarm red/blue lab — 3×3 agent comparison

A small experiment harness on top of **CybORG DroneSwarm (CAGE Challenge 3)** that
compares **3 attacker types × 3 defender types** on the same simulator:

|  | `rule` | `llm` | `rl` |
| --- | --- | --- | --- |
| meaning | hand-written heuristic / FSM | LLM picks from a MITRE menu | frozen tabular Q-policy |
| baseline note | the CAGE-4 report's strong baseline | offline stub by default, Claude if a key is set | trained & cached by `agents/rl.py` |

A synthetic fleet of UAVs + ground vehicles drives the network sim, covering **comms
jamming, GPS spoofing and firmware-worm propagation**. Every (attack × defense)
matchup is logged, scored and visualised, and summarised in a 3×3 grid. See
[docs/architecture.md](docs/architecture.md) for the design and
[docs/demo.md](docs/demo.md) for the per-step animation.

## Folder structure

```
lab/
├─ README.md
├─ requirements.txt          # pinned deps (CybORG installed separately, see below)
├─ src/
│  ├─ sweep.py               # the full 3×3 sweep (main entry point)
│  ├─ run.py                 # one matchup (red_type vs blue_type)
│  ├─ analyze.py             # aggregate all runs -> results/summary.csv
│  ├─ make_dataset.py        # bulk synthetic dataset (no sim)
│  ├─ agents/                # the agent intelligence
│  │  ├─ actions.py          #   MITRE-tagged action catalog (single source of truth)
│  │  ├─ brains.py           #   rule/llm/rl agents for red and blue
│  │  ├─ llm.py              #   offline stub + optional Claude backend
│  │  └─ rl.py               #   tabular Q-learning (train + cache)
│  ├─ sim/                   # environment
│  │  ├─ fleet.py            #   synthetic UAV+UGV telemetry + attacks
│  │  └─ defense.py          #   telemetry detector + response (scored)
│  ├─ viz/                   # visualisation
│  │  ├─ plot.py             #   static figures
│  │  └─ render.py           #   per-step pygame animation / GIF
│  └─ configs/*.yaml         # sweep.yaml + legacy scenario presets
├─ docs/                     # architecture diagram, demo notes, example_animation.gif
├─ results/                  # generated (gitignored): per-matchup runs + grids
└─ data/                     # generated (gitignored): bulk synthetic datasets
```

## Install

Python 3.11, CPU only.

```bash
python -m venv venv
venv\Scripts\activate                       # Windows (Linux/mac: source venv/bin/activate)

# 1) CybORG CC3 (not on PyPI)
git clone https://github.com/cage-challenge/CybORG
pip install -e ./CybORG --no-deps           # --no-deps so it doesn't pull numpy 2.x

# 2) pinned dependencies
pip install -r requirements.txt
```

### Environment variables

| var | effect |
| --- | --- |
| `ANTHROPIC_API_KEY` | if set, the `llm` agent type calls Claude (`claude-haiku-4-5`); otherwise a deterministic offline stub is used so everything runs with no network |
| `LAB_LLM_MODEL` | override the Claude model id (default `claude-haiku-4-5-20251001`) |
| `SDL_VIDEODRIVER=dummy` | run pygame headless (auto-set by `sweep.py` for GIF export) |

## Run

```bash
# full attack(3) × defense(3) comparison  -> results/sweep_*/
python src/agents/rl.py train src/configs/sweep.yaml --episodes 200   # train+cache rl (auto if missing)
python src/sweep.py src/configs/sweep.yaml                            # 9 matchups + grids + per-matchup GIFs
python src/sweep.py src/configs/sweep.yaml --seeds 1 --steps 20 --episodes 20 --fresh   # quick smoke
python src/sweep.py src/configs/sweep.yaml --no-gif                   # skip GIF export (faster)

# one matchup (red/blue each in {rule, llm, rl})
python src/run.py src/configs/sweep.yaml --red rule --blue rl         # -> results/<run_id>/
python src/viz/plot.py <run_id>                                       # static figures
python src/viz/render.py <run_id>                                     # live pygame view  (--gif / --smoke)

python src/analyze.py                                                 # aggregate all runs -> results/summary.csv
```

## Scenario config

```yaml
name: sweep
fleet: {n_uav: 12, n_ugv: 6, grid: 100, max_link: 40}
steps: 40
seeds: [0, 1, 2]
red_types:  [rule, llm, rl]      # agent types swept by sweep.py
blue_types: [rule, llm, rl]
sim:   {starting_num_red: 3, red_spawn_rate: 0.20, max_length_data_links: 40}
defense: {detector: multisensor, snr_thresh: 6, gps_thresh: 8, response: safe_mode}
attacks:
  - {type: jam,        targets: [0, 1, 2, 3], t: [10, 25], drop: 22}   # ids 0..n_uav-1 = UAV
  - {type: gps_spoof,  targets: [12, 13],     t: [15, 35], drift: 3.0} #     n_uav.. = UGV
```

| field | options |
| --- | --- |
| `red_types` / `blue_types` | any of `rule`, `llm`, `rl` |
| `defense.detector` | `none`, `threshold` (jam only), `multisensor` (also GPS); both also run the Canary worm detector |
| `defense.response` | `none`, `safe_mode` (correct spoofed position), `isolate` |
| `defense.canary_recall` / `canary_fp` | Canary worm-detector sensitivity (default `0.8` / `0.03`) |
| `attacks[].type` | `jam`, `gps_spoof` |

`sweep.py` reads `red_types`/`blue_types`; single `run.py` takes `--red`/`--blue`.
Presets in `src/configs/`: `sweep` + legacy `baseline`, `combined`, `passive_blue`,
`defended`, `fsm_red` (run via `python src/run.py src/configs/<name>.yaml --red rule --blue rule`).

## Action catalog (MITRE-tagged)

All agent types choose from one catalog (`src/agents/actions.py`), grounded in real
DroneSwarm primitives + the synthetic `sim/defense.py` layer:

- **Red (~11, MITRE ATT&CK):** Sleep, DiscoverDrones (T1018), Exploit
  Nearest/Random/Farthest (T1210), SeizeControl (T1078), SpreadWorm (T1021),
  Jam Nearest/Farthest (T1498/T1499), BlockComms (T1565), Persist (T1542 firmware).
- **Blue (~13, MITRE D3FEND):** Sleep, Monitor, Analyse, RemoveSessions,
  Retake Suspicious/Random, BlockSuspicious, AllowTraffic, DeployDecoy (deception,
  CC2-winning tactic) — chosen per step — plus DetectJam, DetectGPS, SafeMode,
  Isolate (passive telemetry defences set via the `defense` block, scored by
  `sim/defense.py`, which also runs the **Canary** worm/compromise detector → `comp_F1`).

## Outputs

Per matchup `results/sweep_*/<red>_vs_<blue>/`: `log.csv`, `arrays.npz`, `meta.json`,
and `figs/` (`g_compromise_curve`, `a_reward`, `animation.gif`, `b_compromise`,
`c_snr_jam`, `d_gps_spoof`, `e_fleet_map`, `f_defense`). Top-level `results/sweep_*/`
adds `summary.csv`, `grid_heatmaps.png`, `grid_curves.png`.

**Quantitative metrics** (mean over seeds, in `meta.json`/`summary.csv`):
attack = `final_compromise`, `peak_compromise`, `time_to_first_compromise`,
`compromise_auc`, `recovered`; defense = `blue_reward_total`, `comp_F1` (Canary
worm detection), `jam_F1`, `gps_F1`, `gps_err_before→after`.

## Extending

- **New red attacker / blue manoeuvre** — add a catalog entry in `src/agents/actions.py`
  and a branch in `src/agents/brains.py` (the `_Red` subclasses / `blue_decide`).
- **New detector / response** — extend `run_defense()` (`src/sim/defense.py`); it's
  scored against ground-truth labels automatically.
- **New attack type** — add a branch in `generate_fleet()` (`src/sim/fleet.py`) + a label array.
- **New scenario** — copy a YAML in `src/configs/`.

## Datasets

For offline detector work, dump a labelled synthetic dataset (no sim, so many seeds):

```bash
python src/make_dataset.py src/configs/combined.yaml --seeds 200   # -> data/combined.csv
```

## Limitations

Jamming and GPS spoofing are geometric/signal abstractions, not RF or navigation
physics — only the firmware worm is real CybORG network dynamics. The jam/GPS
detection F1 is therefore config-driven and constant across agent types by design; the
agent comparison happens on the sim network channel.
