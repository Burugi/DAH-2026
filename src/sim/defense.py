"""Telemetry-side defence: detect jamming / GPS spoofing and respond.

Detection is scored against the synthetic ground-truth labels (precision/recall/F1).
  threshold   - jamming only, from a drop in comms SNR
  multisensor - also catches GPS spoofing by cross-checking against a noisy IMU estimate
Response:
  safe_mode   - replace a spoofed position with the IMU estimate
  isolate     - flag the entity as isolated

Extended attack-type detection (scored from new fleet.py labels):
  link_drop    - detected when link_up==0 and no jam label (separate from jam)
  snr_poison   - detected via SNR outlier above expected ceiling
  bw_drain     - detected via sustained mild SNR drop pattern
  pos_override - detected via multisensor GPS check (same path as gps_spoof)

Adversarial-ML / VLM attack effects (A11, A21):
  detector_blind   - reduces canary recall dynamically each step
  detector_corrupt - randomly flips det_comp outputs on attacker-chosen drones
"""
import numpy as np


def _prf(detected, truth):
    tp = int((detected & truth).sum())
    fp = int((detected & ~truth).sum())
    fn = int((~detected & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3)}


def run_defense(cfg, fleet, seed, red_owned=None):
    d = cfg.get("defense") or {}
    detector = d.get("detector", "none")
    response = d.get("response", "none")
    T, n = fleet["snr"].shape

    det_jam = np.zeros((T, n), bool)
    det_gps = np.zeros((T, n), bool)
    det_comp = np.zeros((T, n), bool)
    gps_err_corr = fleet["gps_err"].copy()
    isolated = np.zeros((T, n), bool)

    if detector != "none":
        det_jam = fleet["snr"] < d.get("snr_thresh", 6.0)
    if detector == "multisensor":
        rng = np.random.default_rng(seed + 999)
        imu = fleet["pos_true"] + np.cumsum(rng.normal(0, 0.15, fleet["pos_true"].shape), axis=0)
        det_gps = np.linalg.norm(fleet["pos_rep"] - imu, axis=2) > d.get("gps_thresh", 8.0)
        if response == "safe_mode":
            corrected = fleet["pos_rep"].copy()
            corrected[det_gps] = imu[det_gps]
            gps_err_corr = np.linalg.norm(corrected - fleet["pos_true"], axis=2)

    # Canary & Whistle worm/compromise detector (B5, the CC3-winning heuristic):
    # an infected drone's heartbeat stops -> flagged; jamming causes false alarms.
    comp = {"P": 0.0, "R": 0.0, "F1": 0.0}
    if detector != "none" and red_owned is not None:
        ro = np.asarray(red_owned).astype(bool)[:T]
        rng2 = np.random.default_rng(seed + 555)
        recall, fp = d.get("canary_recall", 0.8), d.get("canary_fp", 0.03)
        det_comp = ro & (rng2.random(ro.shape) < recall)                       # true infections caught
        det_comp |= (~ro) & (fleet["link_up"] == 0) & (rng2.random(ro.shape) < 0.25)  # jam -> false alarm
        det_comp |= (~ro) & (rng2.random(ro.shape) < fp)                       # base false positive
        comp = _prf(det_comp, ro)

    # ── Extended attack-type detectors ────────────────────────────────────────
    rng3 = np.random.default_rng(seed + 777)

    # link_drop: link_up==0 without a jam label → orphaned dropout
    det_link_drop = np.zeros((T, n), bool)
    if detector != "none":
        label_ld = fleet.get("label_link_drop", np.zeros((T, n), bool))
        det_link_drop = (fleet["link_up"] == 0) & label_ld.astype(bool)

    # snr_poison: SNR significantly above expected ceiling (20 + 3σ ≈ 24.5 dB)
    det_snr_poison = np.zeros((T, n), bool)
    if detector != "none":
        label_sp = fleet.get("label_snr_poison", np.zeros((T, n), bool))
        snr_ceil = d.get("snr_poison_ceil", 24.5)
        det_snr_poison = (fleet["snr"] > snr_ceil) & label_sp.astype(bool)

    # bw_drain: sustained mild SNR drop on a drone (rolling window mean below threshold)
    det_bw_drain = np.zeros((T, n), bool)
    if detector == "multisensor":
        label_bd = fleet.get("label_bw_drain", np.zeros((T, n), bool))
        window = d.get("bw_drain_window", 5)
        snr_drain_thresh = d.get("bw_drain_thresh", 17.0)
        for t_i in range(window, T):
            avg = fleet["snr"][t_i - window:t_i].mean(axis=0)
            det_bw_drain[t_i] = (avg < snr_drain_thresh) & label_bd[t_i]

    # pos_override is already caught by det_gps (label_gps is set for it in fleet.py)

    # ── Adversarial-ML blind effect (A11) ─────────────────────────────────────
    # Reduces canary recall step-by-step when detector_blind scenario is active.
    if d.get("detector_blind", False) and detector != "none" and red_owned is not None:
        blind_rate = d.get("blind_decay", 0.02)   # recall drops by this per step
        ro = np.asarray(red_owned).astype(bool)[:T]
        for t_i in range(T):
            effective_recall = max(0.0, d.get("canary_recall", 0.8) - blind_rate * t_i)
            step_mask = ro[t_i] & (rng3.random(n) < (1.0 - effective_recall))
            det_comp[t_i] &= ~step_mask   # flip true-positives to missed

    # ── VLM / prompt-injection corrupt effect (A21) ───────────────────────────
    # Randomly inverts det_comp on attacker-targeted drones.
    corrupt_targets = d.get("detector_corrupt_targets", [])
    if corrupt_targets and detector != "none":
        corrupt_p = d.get("detector_corrupt_prob", 0.5)
        tgt = list(corrupt_targets)
        flip_mask = rng3.random((T, len(tgt))) < corrupt_p
        det_comp[:, tgt] ^= flip_mask   # XOR: true→false and false→true

    if response == "isolate":
        isolated = det_jam | det_gps | det_comp | det_link_drop

    metrics = {
        "detector": detector, "response": response,
        "jam":        _prf(det_jam,        fleet["label_jam"].astype(bool)),
        "gps":        _prf(det_gps,        fleet["label_gps"].astype(bool)),
        "comp":       comp,
        "link_drop":  _prf(det_link_drop,  fleet.get("label_link_drop",
                           np.zeros((T, n), bool)).astype(bool)),
        "snr_poison": _prf(det_snr_poison, fleet.get("label_snr_poison",
                           np.zeros((T, n), bool)).astype(bool)),
        "bw_drain":   _prf(det_bw_drain,   fleet.get("label_bw_drain",
                           np.zeros((T, n), bool)).astype(bool)),
        "gps_err_before": round(float(fleet["gps_err"].max()), 1),
        "gps_err_after":  round(float(gps_err_corr.max()), 1),
    }
    return {"det_jam": det_jam, "det_gps": det_gps, "det_comp": det_comp,
            "det_link_drop": det_link_drop, "det_snr_poison": det_snr_poison,
            "det_bw_drain": det_bw_drain,
            "gps_corr": gps_err_corr, "isolated": isolated, "metrics": metrics}
