"""Figures for the follow-up sections of the blog post (risk-aware CEC, collisions between samples, and the
path-aware risk), in the style of blog_figs.py.

Run after followup_analysis.py:  python followup_figs.py -> results/figs/blog/
"""
import json
import os
import pickle

import numpy as np
from scipy.stats import ncx2

import common as C
from followup_analysis import arc_points, gap_crossings, path_clearance_per_step
from blog_figs import OUT, BLUE, DGRAY, OBST_EDGE, OBST_FILL, line, save, plt
from analyze import RES

ORANGE = "#D26A2C"  # risk-aware CEC and the hybrid
DBLUE = "#2D5BB9"  # tabular GPI with the path-aware risk
STYLE = {  # color, line style, marker face ("tint" = filled with a tint, "white" = open)
    "CEC": (DGRAY, "-", None), "CEC-path": (DGRAY, "-", None),
    "GPI": (BLUE, "-", None), "GPI-sampled": (BLUE, "--", "white"),
    "GPI-path": (DBLUE, "-", None),
    "RiskCEC-soft": (ORANGE, ":", "white"), "RiskCEC-hard": (ORANGE, "--", "white"),
    "RiskCEC-V": (ORANGE, "-", None), "RiskCEC-path-V": (ORANGE, "--", "white"), "Hybrid": (ORANGE, "-", None),
}
NAME = {
    "CEC": "CEC", "CEC-path": "CEC", "GPI": "Tabular GPI", "GPI-path": "GPI, path risk",
    "RiskCEC-soft": "Risk-aware CEC", "RiskCEC-hard": "+ constraints", "RiskCEC-V": "+ GPI value",
    "RiskCEC-path-V": "Risk-aware CEC + GPI value", "Hybrid": "+ GPI plan as a start",
}


def summary():
    return json.load(open(os.path.join(RES, "followup_summary.json")))


def rows_of(suite):
    with open(os.path.join(RES, f"rollouts_{suite}.pkl"), "rb") as fh:
        return pickle.load(fh)


def pick(rows, controller, k, seeds=None):
    return [r for r in rows if r["controller"] == controller and r["noise_scale"] == k
            and (seeds is None or r["seed"] in seeds)]


def styled(ax, x, y, key, **kw):
    c, ls, mfc = STYLE[key]
    if mfc == "white":
        kw.setdefault("mfc", "white")
    return line(ax, x, y, c, ls=ls, **kw)


def sweep(sec, controller):
    return sorted([r for r in sec if r["controller"] == controller and r["k"] > 0], key=lambda r: r["k"])


def ci(r, key):
    return r[f"{key}_ci"] if r["n"] > 1 else (r[key], r[key])


def noise_ticks(ax):
    ax.set_xticks([0.25, 0.5, 1, 1.5, 2])
    ax.set_xticklabels(["0.25", "0.5", "1", "1.5", "2"])


def obstacles(ax, which=range(4), labels=True):
    for i in which:
        x, y, r = C.OBSTACLES[i]
        ax.add_patch(plt.Circle((x, y), r, color=OBST_FILL, lw=0, zorder=1))
        ax.add_patch(plt.Circle((x, y), r + C.ROBOT_RADIUS, fill=False, ec=OBST_EDGE, lw=1.0, ls="--", zorder=1))
        if labels:
            ax.text(x, y, f"C{i + 1}", ha="center", va="center", fontsize=13, color="#666666", zorder=4)


def reference(ax, t0=0, t1=C.T_PERIOD):
    r = C.ref(np.arange(t0, t1 + 1))
    ax.plot(r[:, 0], r[:, 1], color="black", lw=1.0, ls=":", zorder=2)


# ------------------------------------------------------------------------------------------
def fig_riskcec_failures(s, risk_rows, seeds=range(20)):
    """(a) the unconstrained risk-aware CEC plans through C1 at noise x0.25; (b) why: the probability
    is flat outside a thin band; (c) at noise x1 it stalls and falls behind."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw=dict(width_ratios=[1, 1.15, 1.35]))
    ax = axes[0]
    ax.set_aspect("equal")
    obstacles(ax, (0, 2))
    reference(ax)
    for r in pick(risk_rows, "RiskCEC-soft", 0.25, seeds):
        X = r["traj"][:60]
        ax.plot(X[:, 0], X[:, 1], color=ORANGE, lw=1.0, alpha=0.6, zorder=3)
    ax.set_xlim(0.6, 3.1)
    ax.set_ylim(-0.4, 2.2)
    ax.set_title(r"Noise $\times 0.25$: Through C1")

    ax = axes[1]
    sd = np.linspace(-0.2, 0.2, 801)
    radius = C.OBSTACLES[0, 2] + C.ROBOT_RADIUS
    for sig, ls, lab in ((0.01, "-", r"$\sigma$ = 1 cm"), (0.04, "--", r"$\sigma$ = 4 cm")):
        p = ncx2.cdf((radius / sig) ** 2, 2, ((radius + sd) / sig) ** 2)
        ax.plot(100 * sd, p, color="black", ls=ls, lw=1.8, label=lab)
    ax.axvspan(-3, 3, color="#EEEEEE", lw=0, zorder=0)
    a, b = 8.0, -9.0  # two consecutive predicted positions, 17 cm apart
    for d in (a, b):
        p = ncx2.cdf((radius / 0.01) ** 2, 2, ((radius + d / 100) / 0.01) ** 2)
        ax.plot([d], [p], "o", ms=10, mfc=ORANGE, mec=ORANGE, zorder=5)
    ax.annotate("", xy=(b + 0.6, 0.93), xytext=(a - 0.6, 0.07),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3))
    ax.text(-8.5, 0.42, "one step", ha="left", va="center", fontsize=12, color="#444444")
    ax.text(0, -0.06, r"$\pm 3\sigma$ band", ha="center", va="bottom", fontsize=11, color="#555555")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-0.1, 1.08)
    ax.set_xlabel("Distance Outside the Inflated Obstacle (cm)")
    ax.set_title("Collision Probability of a Sample")
    ax.legend(loc="upper right", fontsize=12, handlelength=2.2)

    ax = axes[2]
    ts = s["riskcec_error_k1"]
    t = np.arange(len(ts["GPI"]["mean"]))
    hs = []
    for key in ("RiskCEC-soft", "RiskCEC-hard", "RiskCEC-V", "GPI"):
        hs.append(styled(ax, t, 100 * np.array(ts[key]["mean"]), key, markevery=24, ms=7))
    ax.set_xlim(0, 240)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.set_xlabel("Time Step")
    ax.set_title(r"Mean Tracking Error (cm), Noise $\times 1$")
    ax.set_ylim(bottom=0)
    save(fig, "riskcec_failures.png", rect=(0, 0.1, 1, 1),
         legend=(hs, ["Risk-aware CEC", "+ constraints", "+ GPI value", "Tabular GPI"], "Controller", 0.0, 0.58))


def fig_gap_dash(s, rows_k, k=0.5, seeds=range(40)):
    """(a) GPI's samples straddle the C2-C4 gap while the path goes through it; (b) sampled vs path rates."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ax.set_aspect("equal")
    obstacles(ax, (1, 3), labels=False)
    ax.text(-2.12, -0.95, "C2", ha="center", va="center", fontsize=13, color="#666666", zorder=4)
    ax.text(-1.22, -0.08, "C4", ha="center", va="center", fontsize=13, color="#666666", zorder=4)
    mid = (C.OBSTACLES[1, :2] + C.OBSTACLES[3, :2]) / 2
    ax.annotate("5 cm gap", mid, xytext=(-1.5, -0.25), fontsize=12, color="#444444", ha="left",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1), zorder=6)
    shown = 0
    for r in pick(rows_k, "GPI", k, seeds):
        cross = [c["t"] for c in gap_crossings(r["traj"], r["controls"]) if c["gap"] == "C2-C4"]
        if not cross:
            continue
        t = cross[0]
        px, py = arc_points(r["traj"], r["controls"])
        seg = slice(max(t - 1, 0), t + 2)
        for i in range(seg.start, seg.stop):
            pts = np.stack([px[i], py[i]], -1)
            inside = C.clearance(pts) < 0
            ax.plot(px[i], py[i], color=BLUE, lw=1.1, alpha=0.8, zorder=3)
            ax.plot(px[i][inside], py[i][inside], color="black", lw=2.2, zorder=4)
        X = r["traj"][seg.start: seg.stop + 1]
        ax.plot(X[:, 0], X[:, 1], "o", ms=6, mfc=BLUE, mec="white", mew=0.8, zorder=5, ls="none")
        shown += 1
        if shown >= 12:
            break
    ax.set_xlim(-2.25, -1.05)
    ax.set_ylim(-1.15, 0.25)
    ax.set_title(rf"Tabular GPI at the C2-C4 Gap, Noise $\times {k:g}$", fontsize=15)

    ax = axes[1]
    base = s["baseline"]
    hs = []
    g = sweep(base, "GPI")
    kk = [r["k"] for r in g]
    hs.append(styled(ax, kk, [100 * r["p_coll"] for r in g], "GPI-sampled"))
    hs.append(styled(ax, kk, [100 * r["p_path"] for r in g], "GPI"))
    ax.fill_between(kk, [100 * r["p_path_ci"][0] for r in g], [100 * r["p_path_ci"][1] for r in g],
                    color=BLUE, alpha=0.15, lw=0)
    c = sweep(base, "CEC")
    hs.append(styled(ax, [r["k"] for r in c], [100 * r["p_path"] for r in c], "CEC"))
    ax.set_xlabel(r"Noise Scale ($\times\sigma$)")
    noise_ticks(ax)
    ax.set_ylim(-5, 105)
    ax.set_title("Collision Rate (%)")
    save(fig, "gap_dash.png", rect=(0, 0.1, 1, 1),
         legend=(hs, ["GPI, samples only", "GPI, along the path", "CEC, along the path"], "", 0.0, 0.55))


def fig_path_sweep(s):
    """Tracking error, path collision rate and steps until the first path collision versus noise."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    series = [("CEC-path", s["path"]), ("GPI", s["baseline"]), ("GPI-path", s["path"]), ("Hybrid", s["path"])]
    hs = []
    for key, sec in series:
        g = sweep(sec, key)
        kk = np.array([r["k"] for r in g])
        c = STYLE[key][0]
        hs.append(styled(axes[0], kk, [100 * r["pos_err"] for r in g], key))
        styled(axes[1], kk, [100 * r["p_path"] for r in g], key)
        axes[1].fill_between(kk, [100 * ci(r, "p_path")[0] for r in g], [100 * ci(r, "p_path")[1] for r in g],
                             color=c, alpha=0.12, lw=0)
        styled(axes[2], kk, [r["path_safe_steps_mean"] for r in g], key)
    axes[0].set_title("Tracking Error (cm)")
    axes[1].set_title("Collision Rate Along the Path (%)")
    axes[2].set_title("Steps Until First Collision")
    axes[1].set_ylim(-5, 105)
    axes[2].set_ylim(-5, 255)
    for ax in axes:
        ax.set_xlabel(r"Noise Scale ($\times\sigma$)")
        noise_ticks(ax)
    save(fig, "path_sweep.png", legend=(hs, ["CEC", "GPI, sample risk", "GPI, path risk", "Hybrid"], "Controller"))


def fig_funnel(path_rows, k=1.0, seeds=range(20)):
    """The left lobe: without the GPI plan, the risk-aware CEC follows the reference into the dead end
    between C2 and C4; with it, it turns back at the top like GPI."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    for ax, key in zip(axes, ("RiskCEC-path-V", "Hybrid", "GPI-path")):
        ax.set_aspect("equal")
        obstacles(ax, (1, 3))
        reference(ax, 50, 100)
        col = STYLE[key][0]
        for r in pick(path_rows, key, k, seeds):
            X = r["traj"][40:101]
            ax.plot(X[:, 0], X[:, 1], color=col, lw=1.0, alpha=0.55, zorder=3)
            clr = path_clearance_per_step(r["traj"], r["controls"])[40:100]
            bad = np.flatnonzero(clr < 0)
            if len(bad):
                px, py = arc_points(r["traj"][40:101], r["controls"][40:100])
                i = bad[0]
                j = int(np.argmin(C.clearance(np.stack([px[i], py[i]], -1))))
                ax.plot([px[i, j]], [py[i, j]], "x", color="black", ms=10, mew=2.2, zorder=6)
        ax.set_xlim(-2.75, 0.35)
        ax.set_ylim(-2.25, 2.25)
        ax.set_title({"RiskCEC-path-V": "Risk-aware CEC + GPI Value", "Hybrid": "+ GPI Plan as a Start",
                      "GPI-path": "Tabular GPI, Path Risk"}[key])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "funnel.png"), dpi=200)
    plt.close(fig)


def fig_path_frontier(s):
    """Path collision rate versus tracking error at noise x1 for each family of controllers."""
    cec = sorted([r for r in s["path"] if r["controller"] == "CEC-path" and r["k"] == 1.0]
                 + [r for r in s["path_frontier"] if r["controller"].startswith("CEC-path-m")],
                 key=lambda r: r["pos_err"])
    gpi = sorted(s["baseline_lambda"], key=lambda r: r["pos_err"])
    gpath = sorted([r for r in s["path"] if r["controller"] == "GPI-path" and r["k"] == 1.0]
                   + [r for r in s["path_frontier"] if r["controller"].startswith("GPI-path-lam")],
                   key=lambda r: r["pos_err"])
    hyb = sorted([r for r in s["path"] if r["controller"] == "Hybrid" and r["k"] == 1.0]
                 + [r for r in s["path_frontier"] if r["controller"].startswith("Hybrid-lam")],
                 key=lambda r: r["pos_err"])
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    hs = []
    for key, g in (("CEC-path", cec), ("GPI-sampled", gpi), ("GPI-path", gpath), ("Hybrid", hyb)):
        hs.append(styled(ax, [100 * r["pos_err"] for r in g], [100 * r["p_path"] for r in g], key))

    def lam_of(r):
        name = r["controller"]
        return int(name.split("lam")[1]) if "lam" in name else 1000

    def note(r, text, dx, dy):
        ax.annotate(text, (100 * r["pos_err"], 100 * r["p_path"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=12, color="#444444")

    offsets = {("GPI-path", 1000): (-6, -22), ("GPI-path", 10000): (-60, 10),
               ("Hybrid", 1000): (-30, -22), ("Hybrid", 10000): (-40, -24), ("GPI-sampled", 1000): (10, 2)}
    for key, g in (("GPI-path", gpath), ("Hybrid", hyb), ("GPI-sampled", gpi)):
        for r in g:
            lam = lam_of(r) if key != "GPI-sampled" else int(r["controller"].replace("GPI-lam", ""))
            if (key, lam) in offsets:
                note(r, r"$\lambda$=" + f"{lam:,}", *offsets[(key, lam)])
    for r in cec:
        m = 0.0 if r["controller"] == "CEC-path" else float(r["controller"].replace("CEC-path-m", ""))
        if m >= 0.05:
            note(r, f"m={m * 100:g}cm", -24, 9)
    ax.annotate("$\\lambda$ = 10, 100\nm $\\leq$ 2.5 cm", (11.3, 96), xytext=(7.4, 66), fontsize=12, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(7, 130)
    ax.set_ylim(-14, 110)
    ax.set_xlabel("Tracking Error (cm, log)")
    ax.set_ylabel("Collision Rate Along the Path (%)")
    ax.set_title(r"Safety vs. Tracking at Noise $\times 1$, Checked Along the Path")
    save(fig, "path_frontier.png", rect=(0, 0.1, 1, 1),
         legend=(hs, ["CEC + margin", "GPI, sample risk", "GPI, path risk", "Hybrid"], "", 0.0, 0.52))


if __name__ == "__main__":
    s = summary()
    risk_rows = rows_of("riskcec")
    fig_riskcec_failures(s, risk_rows)
    base_k05 = rows_of("revised_part_k0p5")
    fig_gap_dash(s, base_k05)
    if "path" in s:
        fig_path_sweep(s)
        fig_funnel(rows_of("path"))
    if "path_frontier" in s:
        fig_path_frontier(s)
    print(sorted(os.listdir(OUT)))
