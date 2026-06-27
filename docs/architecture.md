# Architecture

A small experiment harness on top of **CybORG DroneSwarm (CAGE Challenge 3)** that
compares **3 attacker types × 3 defender types** (`rule` / `llm` / `rl`) on the same
simulator, with quantitative metrics and per-step visualisation for each of the 9
matchups.

## Two-channel design

CybORG models a drone network (sessions, exploits, comms links) but has **no GPS/RF
physics**. So the harness runs two aligned channels over one shared fleet:

| channel | produced by | signals |
| --- | --- | --- |
| network compromise (exploit / seize / worm) | the CybORG sim | `red_owned`, `reward` |
| position / GPS / comms (jamming, spoofing) | synthetic layer (`fleet.py`) | `snr`, `link_up`, `gps_err`, attack labels |

The synthetic fleet's initial layout seeds the sim (`starting_positions`), so both
channels describe one fleet. The **agent-type comparison happens on the sim network
channel**; jamming/GPS-spoofing are synthetic scenario context, and the passive
telemetry defences are scored against ground-truth labels (`defense.py`).

## Components

```
                 configs/sweep.yaml
                        │
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │  sweep.py   loops 9 matchups (red_type × blue_type)       │
   └───────┬───────────────────────────────────────┬──────────┘
           │ ensure_trained()                       │ per matchup
           ▼                                        ▼
        rl.py  ──trains──► results/rl_*.npz   run.rollout(cfg, seed, red, blue)
       (tabular Q)                                  │
                                                    ├─ build_env() ─► CybORG DroneSwarm sim
   actions.py  (MITRE catalog)                      │                 + synthetic fleet (fleet.py)
        ▲   ▲                                       │
        │   │                                       ├─ red  = brains.RED_BRAINS[red_type]   (BaseAgent)
   brains.py │  ◄── llm.py (stub / Claude)          ├─ blue = brains.blue_decide(blue_type) (selector)
  (RuleRed/LLMRed/RLRed,                            └─ defense.py  (detect + respond, scored)
   blue_decide rule/llm/rl)                              │
                                                         ▼
                                  run.save_run() ─► results/<run>/ {log.csv, arrays.npz, meta.json}
                                                         │
                              ┌──────────────────────────┼───────────────────────────┐
                              ▼                           ▼                           ▼
                        plot.make_figs()           render.save_gif()            sweep grid
                        (static figures)        (per-step animation.gif)   (heatmaps + 9-panel)
```

- **`agents/actions.py`** — single source of truth: red ~10 (MITRE ATT&CK) and blue ~12
  (MITRE D3FEND) actions, each mapped to a real DroneSwarm primitive or a synthetic
  defence, plus the LLM text menus.
- **`agents/brains.py`** — the three agent types for both sides over that one catalog.
- **`agents/llm.py`** — offline deterministic stub by default; Claude when `ANTHROPIC_API_KEY` is set.
- **`agents/rl.py`** — pure-numpy tabular Monte-Carlo Q-learning, trained once and cached.
- **`sim/fleet.py` · `sim/defense.py`** — synthetic UAV+UGV telemetry / scored telemetry defence.
- **`run.py`** — `build_env` + `rollout` + `save_run` + attack/defense metrics.
- **`sweep.py`** — the 3×3 driver; writes `summary.csv`, `grid_heatmaps.png`, `grid_curves.png`.
- **`viz/plot.py` · `viz/render.py`** — static figures / per-step pygame animation.

## The 3×3 matrix

Rows = attacker type, columns = defender type. Each cell is one matchup directory with
its own metrics and figures; the grid figures summarise all nine. Headline finding
(consistent with the CAGE Challenge 4 report): **rule-based defence dominates** — it
contains every attacker type far better than the learned defenders.
