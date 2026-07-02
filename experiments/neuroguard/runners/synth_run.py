# -*- coding: utf-8 -*-
"""Synthetic-channel experiment driver (no CybORG needed).
Runs fleet.py + defense.py over the team configs to produce EVIDENCE:
 - detection F1 (jam/gps) per scenario
 - GPS spoofing correction (safe_mode) effect
 - STEALTH EVASION: low-drift GPS spoof (A3-slow) evades threshold detection  <-- review finding C1
Outputs CSV + PNG figures to C:\\workspace\\DAH2026_exp.
"""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026")
import numpy as np
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fleet import generate_fleet
from defense import run_defense

REPO = r"C:\workspace\DAH-2026"
OUT  = r"C:\workspace\DAH2026_exp"
os.makedirs(OUT, exist_ok=True)

def load(name):
    return yaml.safe_load(open(os.path.join(REPO, "configs", f"scenario_{name}.yaml"), encoding="utf-8"))

# ============ 1) Per-scenario detection summary ============
scenarios = ["baseline", "combined", "defended", "fsm_red", "passive_blue"]
rows = []
print("=== Scenario detection summary (synthetic channel, mean over seeds) ===")
print(f"{'scenario':14} {'detector':12} {'response':10} {'jamF1':>6} {'gpsF1':>6} {'gpsErrB':>8} {'gpsErrA':>8}")
for name in scenarios:
    cfg = load(name); fl = cfg["fleet"]
    jamF1, gpsF1, gB, gA = [], [], [], []
    for seed in cfg["seeds"]:
        fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], seed, fl["grid"],
                               cfg.get("attacks"), fl.get("max_link", 40))
        m = run_defense(cfg, fleet, seed)["metrics"]
        jamF1.append(m["jam"]["F1"]); gpsF1.append(m["gps"]["F1"])
        gB.append(m["gps_err_before"]); gA.append(m["gps_err_after"])
    det  = (cfg.get("defense") or {}).get("detector", "none")
    resp = (cfg.get("defense") or {}).get("response", "none")
    r = [name, det, resp, round(np.mean(jamF1), 3), round(np.mean(gpsF1), 3),
         round(np.mean(gB), 1), round(np.mean(gA), 1)]
    rows.append(r)
    print(f"{r[0]:14} {r[1]:12} {r[2]:10} {r[3]:6} {r[4]:6} {r[5]:8} {r[6]:8}")

with open(os.path.join(OUT, "summary_detection.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["scenario","detector","response","jam_F1","gps_F1","gpsErr_before","gpsErr_after"])
    w.writerows(rows)

# ============ 2) Fig: SNR over time with jamming (defended, seed 0) ============
cfg = load("defended"); fl = cfg["fleet"]
fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], 0, fl["grid"], cfg.get("attacks"), 40)
snr = fleet["snr"]; jam_tgt = [0,1,2,3]
plt.figure(figsize=(7,4))
for e in range(snr.shape[1]):
    if e in jam_tgt: continue
    plt.plot(snr[:, e], color="0.8", lw=0.8)
for e in jam_tgt:
    plt.plot(snr[:, e], color="crimson", lw=1.6, label="jammed UAV" if e==jam_tgt[0] else None)
plt.axhline(6, ls="--", color="navy", label="link threshold (6 dB)")
plt.axvspan(10, 25, color="orange", alpha=0.12, label="jam window t=10-25")
plt.xlabel("step"); plt.ylabel("comms SNR (dB)"); plt.title("S3 Jamming: SNR collapse on targeted UAVs (defended, seed0)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig1_snr_jam.png"), dpi=130); plt.close()

# ============ 3) Fig: GPS spoof correction (safe_mode), spoofed UGVs ============
dfd = run_defense(cfg, fleet, 0)
before = fleet["gps_err"]; after = dfd["gps_corr"]; gps_tgt = [12,13]
plt.figure(figsize=(7,4))
for e in gps_tgt:
    plt.plot(before[:, e], color="crimson", lw=1.6, label="spoofed gps_err (before)" if e==gps_tgt[0] else None)
    plt.plot(after[:, e],  color="green",  lw=1.6, ls="--", label="after safe_mode correction" if e==gps_tgt[0] else None)
plt.axvspan(15, 35, color="orange", alpha=0.12, label="spoof window t=15-35")
plt.xlabel("step"); plt.ylabel("position error (m)"); plt.title("S2 GPS spoof: B2 multisensor+safe_mode restores position (defended, seed0)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig2_gps_correction.png"), dpi=130); plt.close()

# ============ 4) Fig: detection F1 bar — multisensor vs threshold ============
def mean_f1(name):
    cfg = load(name); fl = cfg["fleet"]; J,G = [],[]
    for seed in cfg["seeds"]:
        fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], seed, fl["grid"], cfg.get("attacks"), 40)
        m = run_defense(cfg, fleet, seed)["metrics"]; J.append(m["jam"]["F1"]); G.append(m["gps"]["F1"])
    return np.mean(J), np.mean(G)
dj, dg = mean_f1("defended")   # multisensor
fj, fg = mean_f1("fsm_red")    # threshold (jam only)
x = np.arange(2); w = 0.35
plt.figure(figsize=(6.5,4))
plt.bar(x-w/2, [dj, dg], w, label="defended (multisensor)", color="seagreen")
plt.bar(x+w/2, [fj, fg], w, label="fsm_red (threshold)", color="indianred")
plt.xticks(x, ["jamming F1", "GPS spoof F1"]); plt.ylim(0,1.05); plt.ylabel("detection F1")
plt.title("Detection: single-sensor threshold cannot catch GPS spoof (F1=0)\nmultisensor cross-check needed")
for i,v in enumerate([dj,dg]): plt.text(i-w/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
for i,v in enumerate([fj,fg]): plt.text(i+w/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig3_detection_f1.png"), dpi=130); plt.close()

# ============ 5) Fig: STEALTH EVASION — low drift evades threshold (A3-slow / Raven) ============
# review finding C1: stealth attack staying under the cross-check threshold evades L1 detection
cfg = load("defended"); fl = cfg["fleet"]
drifts = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
recalls, errs = [], []
for d in drifts:
    rr, ee = [], []
    for seed in cfg["seeds"]:
        atk = [{"type":"gps_spoof", "targets":[12,13], "t":[15,35], "drift":d}]
        fleet = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], seed, fl["grid"], atk, 40)
        m = run_defense(cfg, fleet, seed)["metrics"]
        rr.append(m["gps"]["R"]); ee.append(m["gps_err_before"])
    recalls.append(np.mean(rr)); errs.append(np.mean(ee))
plt.figure(figsize=(7,4.2))
plt.plot(drifts, recalls, "o-", color="darkblue", lw=1.8)
plt.axhline(0.5, ls=":", color="gray")
plt.axvspan(0.0, 0.8, color="crimson", alpha=0.10)
plt.text(0.45, 0.08, "EVADES\n(A3-slow / Raven\nstays under threshold)", color="crimson", ha="center", fontsize=8)
plt.text(3.0, 0.92, "caught", color="green", ha="center", fontsize=9)
plt.xlabel("GPS spoof drift per step (m)"); plt.ylabel("detection recall (multisensor, gps_thresh=8)")
plt.title("E2 evidence: low-drift (stealth) GPS spoof evades single-layer threshold detection")
plt.ylim(-0.05, 1.05); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig4_stealth_evasion.png"), dpi=130); plt.close()

print("\n=== Stealth evasion sweep (drift -> mean gps recall) ===")
for d, r, e in zip(drifts, recalls, errs):
    print(f"  drift={d:>3}  recall={r:.2f}  max_gps_err={e:.1f}m   {'<-- evades' if r < 0.5 else ''}")

# ============ 6) Fig: fleet map true vs reported (defended seed0, last spoof step) ============
t = 34
plt.figure(figsize=(5.6,5.6))
pt = fleet0 = generate_fleet(fl["n_uav"], fl["n_ugv"], cfg["steps"], 0, fl["grid"], cfg.get("attacks"), 40)
true = pt["pos_true"][t]; rep = pt["pos_rep"][t]
uav = slice(0, fl["n_uav"]); ugv = slice(fl["n_uav"], pt["n"])
plt.scatter(true[uav,0], true[uav,1], c="steelblue", s=30, label="UAV true")
plt.scatter(true[ugv,0], true[ugv,1], c="darkgreen", marker="s", s=30, label="UGV true")
spoofed = [12,13]
for e in spoofed:
    plt.plot([true[e,0], rep[e,0]], [true[e,1], rep[e,1]], "r-", lw=1)
    plt.scatter(rep[e,0], rep[e,1], c="crimson", marker="x", s=70, label="UGV reported (spoofed)" if e==spoofed[0] else None)
plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.title("S2 GPS spoof: reported UGV positions pulled away from truth (t=34)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig5_fleet_map.png"), dpi=130); plt.close()

print("\nSaved figures + summary_detection.csv to", OUT)
print("figs:", [f for f in os.listdir(OUT) if f.endswith(".png")])
