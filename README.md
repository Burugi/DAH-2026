# DroneSwarm Red/Blue Lab — 3×3 Agent Comparison

A small, local experiment harness that pits **cyber attackers (red) against defenders
(blue)** over a swarm of drones, and answers one question: **which kind of agent wins?**

---

## Overview

The lab runs on **CybORG DroneSwarm (CAGE Challenge 3)** with a synthetic fleet of
**12 UAVs + 6 UGVs** moving on a 2D map. Both the attacker and the defender are built in
**three "agent" styles**, and we play every attacker against every defender, giving a
**3×3 grid of 9 matchups**:

| style | how it decides | one line |
| --- | --- | --- |
| `rule` | hand-written heuristic | fast, interpretable, the CAGE-winning recipe |
| `llm` | a language model picks from a menu | offline stub by default, Claude if a key is set |
| `rl` | a learned tabular Q-policy | trained once, then frozen |

Every action is tagged with a **MITRE** technique (ATT&CK for attack, D3FEND for defense),
so the experiment maps onto real-world threats and mitigations.

**Headline result:** the rule-based style is strongest, and it is clearest on **defense**.
Rule-based defense keeps the fleet safest against every attacker (see the grid below).
This matches the CAGE Challenge 4 finding that heuristics beat learned agents.

```
lab/
├─ README.md          ├─ requirements.txt
├─ src/               # all code
│  ├─ sweep.py run.py analyze.py make_dataset.py
│  ├─ agents/  actions.py brains.py llm.py rl.py
│  ├─ sim/     fleet.py defense.py
│  ├─ viz/     plot.py render.py
│  └─ configs/*.yaml
├─ docs/              # report, architecture, demo GIFs
└─ results/  data/    # generated output (not in git)
```

---

## Quick Start

Python 3.11, CPU only.

```bash
# 1) install CybORG CC3 (not on PyPI) + pinned deps
git clone https://github.com/cage-challenge/CybORG
pip install -e ./CybORG --no-deps
pip install -r requirements.txt

# 2) run the full 3x3 comparison  (trains the rl policy on first run, ~1 min)
python src/sweep.py src/configs/sweep.yaml            # -> results/sweep_*/
python src/analyze.py                                 # summary table of all runs

# one matchup at a time (red/blue each in {rule, llm, rl})
python src/run.py src/configs/sweep.yaml --red rule --blue rl
python src/viz/render.py <run_id>                     # live viewer; add --gif to save an animation
```

Optional environment variables: set `ANTHROPIC_API_KEY` to make the `llm` agent call
Claude (otherwise a deterministic offline stub is used, so everything runs with no
network); `SDL_VIDEODRIVER=dummy` runs the viewer headless for GIF export.

---

## Experiment Settings

**Environment.** 18 entities (12 UAV + 6 UGV) on a 100×100 grid. The simulator has two
channels that share one fleet: the CybORG network channel produces who is **compromised**
and the reward, and a synthetic telemetry channel produces position, signal quality,
**jamming** and **GPS spoofing** with ground-truth labels.

**Action pools.** Both sides choose, each step, from a fixed menu of MITRE-tagged actions
(full list in `src/agents/actions.py`):

- **Attack (red), ~11:** discover, exploit (nearest / random / farthest), seize, spread
  worm, jam, block comms, persist.
- **Defense (blue), ~13:** monitor, analyse, remove sessions, retake, block, allow,
  deploy decoy, plus passive detectors for jamming / GPS / worm and a safe-mode position fix.

**Evaluation.** Each of the 9 matchups is run for **5 random seeds × 40 steps**, then
averaged. We track attack metrics (final compromised fraction, time to first compromise,
spread over time, drones recovered) and defense metrics (cumulative blue reward, worm
detection F1, jamming/GPS detection F1, GPS error before vs after correction). Reading the
3×3 grid is simple: scan a **column** to see how one defender does against all attackers
(a strong defender keeps the whole column low), and scan a **row** to see how one attacker
does against all defenders.

---

## Sample Visualization

The viewer draws the fleet step by step. The colours and shapes mean:

- **Fill colour = team.** Blue is a friendly (defended) drone, red is a compromised drone.
- **Shape = platform.** Triangle is a UAV, square is a UGV.
- **Purple ring = jammed.** **Orange arrow = GPS spoofed** (points from the real position to
  the faked one). **Yellow ring = the defender detected something** on that drone.

**A strong attack vs a weak defense (`rule` attacker, `rl` defender).** The fleet turns red
as the worm spreads and the defender fails to keep up.

![rule attacker vs rl defender](docs/gifs/matchup_rule_vs_rl.gif)

**A rule-based defense holding (`rule` vs `rule`).** Drones that turn red are retaken and go
back to blue, so the fleet stays mostly defended.

![rule attacker vs rule defender](docs/gifs/matchup_rule_vs_rule.gif)

**One attack action in isolation (SeizeControl).** The attacker takes over a drone and the
red footprint grows. We verified every action this way.

![red SeizeControl](docs/gifs/action_red_SeizeControl.gif)

**One defense action in isolation (RetakeSuspicious).** The defender keeps retaking
compromised drones, driving the compromised count to zero.

![blue RetakeSuspicious](docs/gifs/action_blue_RetakeSuspicious.gif)

**The 3×3 result at a glance.** Left is the final compromised fraction (lower is better for
the defender), middle is blue reward, right is worm-detection F1.

![3x3 grid](docs/gifs/grid_heatmaps.png)

Final compromised fraction, averaged over 5 seeds (lower is better for the defender):

| attacker \ defender | rule | llm | rl |
| --- | --- | --- | --- |
| **rule** | 0.36 | 0.57 | 0.86 |
| **llm** | 0.22 | 0.47 | 0.57 |
| **rl** | 0.23 | 0.39 | 0.81 |

The `rule` defender column is the lowest everywhere, so it is the strongest defense. A GIF
for **every** matchup and **every** action, plus per-matchup figures, is regenerated under
`results/` by the commands above (`results/` is not committed to keep the repo light).

---

## More

- **Architecture and design:** `docs/architecture.md`
- **Visualisation details and all commands:** `docs/demo.md`
- **Full write-up (report):** `docs/report.md`
- **Extend it:** add an action in `src/agents/actions.py` and a branch in
  `src/agents/brains.py`; add a detector in `src/sim/defense.py`; add a scenario YAML in
  `src/configs/`.
- **Limitation:** jamming and GPS spoofing are signal/geometry abstractions, not RF or
  navigation physics, so their detection scores depend on the scenario settings. The agent
  comparison itself happens on the real CybORG network channel.
