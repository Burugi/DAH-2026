"""Synthetic mixed fleet (UAVs + ground vehicles) with telemetry and attack labels.

CybORG DroneSwarm has no GPS/RF physics, so positions, comms SNR and the
jamming / GPS-spoofing attacks live here as ground-truth-labelled signals.
The firmware/worm attack is realised by the sim, not here.

All arrays have shape (steps, n) or (steps, n, 2) and align with sim step t.

Supported attack types (attacks list passed to generate_fleet):
  jam               - lower SNR by `drop` dB  (label_jam)
  gps_spoof         - accumulate position drift `drift` m/step  (label_gps)
  link_drop         - force link_up=0 during window  (label_link_drop)
  snr_poison        - artificially raise SNR for attacker-favoured links  (label_snr_poison)
  bandwidth_drain   - continuous mild SNR bleed on compromised drones  (label_bw_drain)
  position_override - hard-set reported position to attacker coords  (label_pos_override)
"""
import numpy as np


def generate_fleet(n_uav=12, n_ugv=6, steps=40, seed=0, grid=100.0,
                   attacks=None, max_link=40.0):
    rng = np.random.default_rng(seed)
    n = n_uav + n_ugv
    types = ["uav"] * n_uav + ["ugv"] * n_ugv

    pos = np.zeros((steps, n, 2))
    pos[0, :n_uav] = rng.uniform(10, grid - 10, size=(n_uav, 2))
    pos[0, n_uav:, 0] = rng.uniform(5, grid - 5, size=n_ugv)
    pos[0, n_uav:, 1] = grid * 0.45 + rng.normal(0, 2.0, size=n_ugv)

    vel = np.zeros((n, 2))
    vel[:n_uav] = rng.uniform(-4, 4, size=(n_uav, 2))      # UAVs roam freely
    vel[n_uav:, 0] = rng.uniform(-1.5, 1.5, size=n_ugv)    # UGVs crawl along a road
    road_y = pos[0, n_uav:, 1].copy()

    for t in range(1, steps):
        step = pos[t - 1] + vel
        vel[step[:, 0] < 0, 0] *= -1
        vel[step[:, 0] > grid, 0] *= -1
        vel[step[:, 1] < 0, 1] *= -1
        vel[step[:, 1] > grid, 1] *= -1
        pos[t] = np.clip(pos[t - 1] + vel, 0, grid)
        vel[:n_uav] = np.clip(vel[:n_uav] + rng.normal(0, 0.6, (n_uav, 2)), -5, 5)
        pos[t, n_uav:, 1] = road_y + rng.normal(0, 0.8, size=n_ugv)   # keep UGVs on lane

    snr = 20.0 + rng.normal(0, 1.5, size=(steps, n))
    pos_rep = pos.copy()
    label_jam          = np.zeros((steps, n), bool)
    label_gps          = np.zeros((steps, n), bool)
    label_link_drop    = np.zeros((steps, n), bool)
    label_snr_poison   = np.zeros((steps, n), bool)
    label_bw_drain     = np.zeros((steps, n), bool)
    label_pos_override = np.zeros((steps, n), bool)

    for atk in attacks or []:
        tgt = list(atk["targets"])
        t0, t1 = max(0, atk["t"][0]), min(steps - 1, atk["t"][1])

        if atk["type"] == "jam":
            snr[t0:t1 + 1][:, tgt] -= atk.get("drop", 22.0)
            label_jam[t0:t1 + 1][:, tgt] = True

        elif atk["type"] == "gps_spoof":
            drift = atk.get("drift", 3.0)
            for k, t in enumerate(range(t0, t1 + 1)):
                pos_rep[t][tgt] = pos[t][tgt] + drift * (k + 1)
            label_gps[t0:t1 + 1][:, tgt] = True

        elif atk["type"] == "link_drop":
            # Force link_up=0 on targets during window (SATCOM dropout / deauth).
            # SNR is driven far below the link_up threshold (6 dB).
            snr[t0:t1 + 1][:, tgt] = np.minimum(
                snr[t0:t1 + 1][:, tgt], atk.get("floor", 1.0)
            )
            label_link_drop[t0:t1 + 1][:, tgt] = True

        elif atk["type"] == "snr_poison":
            # Artificially raise SNR for attacker-chosen drone pairs to make
            # them appear as preferred routing neighbours (Sybil/Byzantine).
            snr[t0:t1 + 1][:, tgt] += atk.get("boost", 15.0)
            label_snr_poison[t0:t1 + 1][:, tgt] = True

        elif atk["type"] == "bandwidth_drain":
            # Continuous mild SNR bleed simulating a covert exfil side-channel
            # consuming bandwidth on compromised drones.
            snr[t0:t1 + 1][:, tgt] -= atk.get("drain", 3.0)
            label_bw_drain[t0:t1 + 1][:, tgt] = True

        elif atk["type"] == "position_override":
            # Hard-set reported position to attacker-specified coordinates each
            # step (GCS/MAVLink command injection: drone flies to wrong target).
            dest = np.array(atk.get("dest", [0.0, 0.0]), dtype=float)
            for t in range(t0, t1 + 1):
                pos_rep[t][tgt] = dest
            label_pos_override[t0:t1 + 1][:, tgt] = True
            label_gps[t0:t1 + 1][:, tgt] = True   # also counted as GPS error

    return {
        "types": types, "n": n, "n_uav": n_uav, "steps": steps, "grid": grid,
        "pos_true": pos, "pos_rep": pos_rep, "snr": snr,
        "link_up": (snr > 6.0).astype(np.int8),
        "gps_err": np.linalg.norm(pos_rep - pos, axis=2),
        "label_jam":          label_jam,
        "label_gps":          label_gps,
        "label_link_drop":    label_link_drop,
        "label_snr_poison":   label_snr_poison,
        "label_bw_drain":     label_bw_drain,
        "label_pos_override": label_pos_override,
    }


def starting_positions(fleet):
    """Initial positions for DroneSwarmScenarioGenerator(starting_positions=...)."""
    g = fleet["grid"]
    return [np.array([int(np.clip(x, 0, g - 1)), int(np.clip(y, 0, g - 1))])
            for x, y in fleet["pos_true"][0]]


def truncate(fleet, t):
    """Cut all arrays to t steps (used when an episode ends early)."""
    for k in ("pos_true", "pos_rep", "snr", "link_up", "gps_err",
              "label_jam", "label_gps", "label_link_drop",
              "label_snr_poison", "label_bw_drain", "label_pos_override"):
        fleet[k] = fleet[k][:t]
    fleet["steps"] = t
    return fleet


if __name__ == "__main__":
    f = generate_fleet(attacks=[{"type": "jam", "targets": [0, 1], "t": [10, 25]},
                                {"type": "gps_spoof", "targets": [12], "t": [15, 35]}])
    print(f"{f['n']} entities ({f['n_uav']} uav), {f['steps']} steps")
    print(f"max gps_err {f['gps_err'].max():.1f}, min snr {f['snr'].min():.1f}, "
          f"jam {f['label_jam'].sum()} gps {f['label_gps'].sum()} entity-steps")
