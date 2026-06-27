"""Static figures for one run: python plot.py <run_id>  -> results/<run_id>/figs/

make_figs(dir) also works on any results directory (used by sweep.py for the
nested per-matchup folders).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


def make_figs(d):
    z = np.load(os.path.join(d, "arrays.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    figs = os.path.join(d, "figs"); os.makedirs(figs, exist_ok=True)
    save = lambda name: (plt.savefig(os.path.join(figs, name), dpi=115, bbox_inches="tight"), plt.close())

    types = list(z["types"]); n = len(types); n_uav = types.count("uav")
    name = meta["config"]["name"]
    rt, bt = meta.get("red_type", ""), meta.get("blue_type", "")
    if rt:
        name = f"{name} [{rt} vs {bt}]"
    reward, red = z["reward"], z["red_owned"]
    snr, gps, ljam, lgps = z["snr"], z["gps_err"], z["label_jam"], z["label_gps"]
    pt, pr = z["pos_true"], z["pos_rep"]
    S, T = reward.shape

    # reward (mean +/- std over seeds)
    cum = np.cumsum(reward, 1); m, sd = cum.mean(0), cum.std(0)
    plt.figure(figsize=(7, 4))
    plt.plot(m, ".-", label=f"mean of {S} seeds")
    plt.fill_between(range(T), m - sd, m + sd, alpha=.25, label="±1 std")
    plt.xlabel("step"); plt.ylabel("cumulative blue reward")
    plt.title(f"{name} — reward"); plt.legend(); plt.grid(alpha=.3); save("a_reward.png")

    # compromised drones over time (attack progression, mean +/- std over seeds)
    comp = red.sum(2)                                     # (seeds, steps)
    cm, csd = comp.mean(0), comp.std(0)
    plt.figure(figsize=(7, 4))
    plt.plot(cm, ".-", color="firebrick", label=f"mean of {S} seeds")
    plt.fill_between(range(T), cm - csd, cm + csd, alpha=.25, color="firebrick", label="±1 std")
    plt.xlabel("step"); plt.ylabel("# compromised drones")
    plt.title(f"{name} — compromise over time"); plt.legend(); plt.grid(alpha=.3)
    save("g_compromise_curve.png")

    # worm compromise heatmap (seed 0)
    plt.figure(figsize=(9, 4.5))
    plt.imshow(red[0].T, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    plt.axhline(n_uav - 0.5, color="navy", ls="--")
    plt.yticks(range(n), [f"{i}:{t}" for i, t in enumerate(types)], fontsize=6)
    plt.xlabel("step"); plt.ylabel("entity"); plt.colorbar(ticks=[0, 1])
    plt.title(f"{name} — worm spread (sim, seed0)"); save("b_compromise.png")

    # comms SNR + jamming (seed 0)
    plt.figure(figsize=(9, 4.5))
    plt.imshow(snr[0].T, aspect="auto", cmap="viridis")
    plt.yticks(range(n), [f"{i}:{t}" for i, t in enumerate(types)], fontsize=6)
    plt.xlabel("step"); plt.ylabel("entity"); plt.colorbar(label="SNR (dB)")
    plt.title(f"{name} — comms SNR + jamming (seed0)"); save("c_snr_jam.png")

    # GPS spoof drift (seed 0)
    plt.figure(figsize=(8, 4))
    spoofed = np.where(lgps[0].any(0))[0]
    for e in spoofed:
        plt.plot(gps[0, :, e], ".-", label=f"entity {e} ({types[e]})")
    plt.xlabel("step"); plt.ylabel("GPS error (cells)")
    plt.title(f"{name} — GPS spoof drift (seed0)"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    save("d_gps_spoof.png")

    # fleet map snapshots
    snaps = sorted({0, T // 2, T - 1})
    grid = meta["config"]["fleet"]["grid"]
    fig, axs = plt.subplots(1, len(snaps), figsize=(5 * len(snaps), 5), squeeze=False)
    for ax, t in zip(axs[0], snaps):
        uav, ugv = range(n_uav), range(n_uav, n)
        ax.scatter(pt[0, t, uav, 0], pt[0, t, uav, 1], c="tab:blue", marker="^", s=60, label="UAV")
        ax.scatter(pt[0, t, ugv, 0], pt[0, t, ugv, 1], c="tab:green", marker="s", s=60, label="UGV")
        for e in np.where(ljam[0, t])[0]:
            ax.scatter(*pt[0, t, e], facecolors="none", edgecolors="red", s=160, lw=2)
        for e in np.where(lgps[0, t])[0]:
            ax.annotate("", xy=pr[0, t, e], xytext=pt[0, t, e],
                        arrowprops=dict(arrowstyle="->", color="orange", lw=1.8))
        for e in np.where(red[0, t])[0]:
            ax.scatter(*pt[0, t, e], facecolors="none", edgecolors="black", s=220, lw=1.5)
        ax.set(xlim=(0, grid), ylim=(0, grid), title=f"step {t}", aspect="equal")
        ax.grid(alpha=.2)
    axs[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{name} — fleet map (red=jam, orange=spoof, black=owned)")
    save("e_fleet_map.png")

    # defence: detection vs truth + GPS mitigation
    dfn = meta.get("defense", {})
    if dfn.get("detector", "none") != "none":
        dj, dg, gc = z["det_jam"][0], z["det_gps"][0], z["gps_corr"][0]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        if "det_comp" in z.files:
            ax[0].plot(red[0].sum(1), color="firebrick", label="compromise (truth)")
            ax[0].plot(z["det_comp"][0].sum(1), "--", color="salmon", label="compromise (detected)")
        ax[0].plot(ljam[0].sum(1), color="tab:blue", label="jam (truth)")
        ax[0].plot(dj.sum(1), "--", color="tab:cyan", label="jam (detected)")
        ax[0].plot(lgps[0].sum(1), color="tab:orange", label="gps (truth)")
        ax[0].plot(dg.sum(1), "--", color="gold", label="gps (detected)")
        ax[0].set(xlabel="step", ylabel="# entities")
        ax[0].set_title(f"detection ({dfn['detector']}): comp F1={dfn.get('comp_F1', '-')} "
                        f"jam F1={dfn['jam_F1']} gps F1={dfn['gps_F1']}")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        for e in spoofed:
            ax[1].plot(gps[0, :, e], color="tab:orange", label="before" if e == spoofed[0] else None)
            ax[1].plot(gc[:, e], color="tab:green", label=f"after ({dfn['response']})" if e == spoofed[0] else None)
        ax[1].set(xlabel="step", ylabel="GPS error (cells)")
        ax[1].set_title(f"GPS mitigation: {dfn['gps_err_before']} -> {dfn['gps_err_after']}")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        save("f_defense.png")
    print(f"figures -> {figs}")


if __name__ == "__main__":
    arg = sys.argv[1]
    make_figs(arg if os.path.isdir(arg) else os.path.join(RESULTS, arg))
