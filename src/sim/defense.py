"""Telemetry-side defence: detect jamming / GPS spoofing and respond.

Detection is scored against the synthetic ground-truth labels (precision/recall/F1).
  threshold   - jamming only, from a drop in comms SNR
  multisensor - also catches GPS spoofing by cross-checking against a noisy IMU estimate
Response:
  safe_mode   - replace a spoofed position with the IMU estimate
  isolate     - flag the entity as isolated
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

    if response == "isolate":
        isolated = det_jam | det_gps | det_comp

    metrics = {
        "detector": detector, "response": response,
        "jam": _prf(det_jam, fleet["label_jam"].astype(bool)),
        "gps": _prf(det_gps, fleet["label_gps"].astype(bool)),
        "comp": comp,
        "gps_err_before": round(float(fleet["gps_err"].max()), 1),
        "gps_err_after": round(float(gps_err_corr.max()), 1),
    }
    return {"det_jam": det_jam, "det_gps": det_gps, "det_comp": det_comp,
            "gps_corr": gps_err_corr, "isolated": isolated, "metrics": metrics}
