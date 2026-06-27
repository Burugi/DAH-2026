"""Aggregate every matchup under results/ into one table: python analyze.py

Walks results/ recursively (so it picks up both single runs and the nested
sweep_*/<red>_vs_<blue>/ folders) and reads the attack/defense metrics that
run.save_run() stored in meta.json.
"""
import os, json, csv

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
COLS = ["path", "scenario", "red_type", "blue_type", "defense",
        "final_compromise", "peak_compromise", "time_to_first_compromise",
        "compromise_auc", "blue_reward_total", "recovered",
        "comp_F1", "jam_F1", "gps_F1", "gps_err_before", "gps_err_after"]


def find_runs(root):
    for dp, _, files in os.walk(root):
        if "meta.json" in files and "arrays.npz" in files:
            yield dp


def main():
    rows = []
    for d in sorted(find_runs(RES)):
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        if "metrics" not in meta:                         # skip legacy-format runs
            continue
        m, dfn = meta["metrics"], meta.get("defense", {})
        rows.append([os.path.relpath(d, RES), meta["config"]["name"],
                     meta.get("red_type", ""), meta.get("blue_type", ""),
                     f"{dfn.get('detector', 'none')}/{dfn.get('response', 'none')}",
                     m["final_compromise"], m["peak_compromise"],
                     m["time_to_first_compromise"], m["compromise_auc"],
                     m["blue_reward_total"], m["recovered"], m.get("comp_F1", 0.0),
                     m["jam_F1"], m["gps_F1"], m["gps_err_before"], m["gps_err_after"]])

    with open(os.path.join(RES, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([COLS] + rows)

    print(f"{'red':6}{'blue':6}{'scenario':12}{'finalComp':11}{'blueR':9}{'compF1':7}{'jamF1':7}{'gpsF1':7}")
    print("-" * 65)
    for r in rows:
        print(f"{r[2]:6}{r[3]:6}{r[1]:12}{r[5]:<11}{r[9]:<9}{r[11]:<7}{r[12]:<7}{r[13]:<7}")
    print(f"\n-> {os.path.join(RES, 'summary.csv')} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
