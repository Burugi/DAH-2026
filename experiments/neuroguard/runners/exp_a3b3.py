# -*- coding: utf-8 -*-
"""#6 A3 <-> B3 coupling (synthetic, no CybORG).
Models MAVLink 2.0 anti-replay: a message carries a timestamp; the receiver rejects it
if it is not newer than the last accepted one. Two receiver designs:
  - gps_time : the freshness reference comes from GPS time -> the attacker spoofs GPS time
               backward (A3), making an OLD captured message look fresh -> REPLAY ACCEPTED.
  - rtc      : an independent monotonic hardware clock -> immune to GPS spoof -> replay rejected.
We sweep the GPS time rollback and measure replay-acceptance (bypass) rate.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = r"C:\workspace\DAH2026_exp"


def bypass_rate(defense, rollback, trials=600, seed0=0):
    acc = 0
    for k in range(trials):
        rng = np.random.default_rng(seed0 + k)
        T = 40.0
        times = np.sort(rng.uniform(0.0, 0.5 * T, size=20))   # legit signed messages (monotonic accepted)
        last = float(times.max())                             # receiver's last-accepted timestamp
        t_cap = float(rng.choice(times[:-1]))                 # attacker captures an OLDER signed message
        if defense == "gps_time":
            eff_last = last - rollback                        # GPS spoof rolls the freshness window back
            accepted = t_cap > eff_last
        else:                                                 # rtc: monotonic, spoof has no effect
            accepted = t_cap > last
        acc += int(accepted)
    return acc / trials


rolls = list(range(0, 26, 2))
gps = [bypass_rate("gps_time", r) for r in rolls]
rtc = [bypass_rate("rtc", r) for r in rolls]

print("=== #6 A3->B3: MAVLink anti-replay bypass via GPS-time spoofing ===")
print(f"{'GPS rollback (s)':18}{'gps-time accept':>16}{'rtc accept':>12}")
for r, g, t in zip(rolls, gps, rtc):
    print(f"{r:18}{g:16.2f}{t:12.2f}")
print(f"\nHeadline: GPS-time anti-replay bypass reaches {max(gps):.0%} as rollback grows; "
      f"RTC stays {max(rtc):.0%} (replay rejected).")

plt.figure(figsize=(7, 4.3))
plt.plot(rolls, gps, "o-", color="crimson", lw=1.9, label="GPS-time anti-replay (B3 default)")
plt.plot(rolls, rtc, "s--", color="seagreen", lw=1.9, label="independent RTC + monotonic seq (B3 fix)")
plt.fill_between(rolls, gps, alpha=0.08, color="crimson")
plt.xlabel("GPS time rollback by attacker (s)  [A3 spoofing]")
plt.ylabel("replay acceptance (bypass) rate")
plt.title("#6 A3 breaks B3: GPS-time spoof bypasses MAVLink anti-replay; RTC blocks it")
plt.ylim(-0.05, 1.05); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig14_a3b3.png"), dpi=130); plt.close()

with open(os.path.join(OUT, "summary_a3b3.csv"), "w", newline="", encoding="utf-8") as f:
    import csv
    w = csv.writer(f); w.writerow(["gps_rollback_s", "gps_time_accept", "rtc_accept"])
    for r, g, t in zip(rolls, gps, rtc): w.writerow([r, round(g, 3), round(t, 3)])
print("Saved fig14_a3b3.png, summary_a3b3.csv")
