"""Figures for the blog post, in a minimal paper style (typewriter font, boxed axes, dashed grid,
blue for tabular GPI and grays for the others).

Run after revised_analysis.py and phase_robustness.py:  python blog_figs.py -> results/figs/blog/
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors

import common as C
from analyze import RES, load

OUT = os.path.join(RES, "figs", "blog")
os.makedirs(OUT, exist_ok=True)

BLUE, DGRAY, LGRAY = "#598BE7", "#7F7F7F", "#B4B4B4"
COL = {"CEC": DGRAY, "GPI": BLUE, "RBF": LGRAY}
PATH_COL = {"CEC": DGRAY, "GPI": BLUE, "RBF": "#8C8C8C"}  # RBF paths slightly darker to stay visible
NAME = {"CEC": "CEC", "GPI": "Tabular GPI", "RBF": "RBF GPI"}
METHODS = ("CEC", "GPI", "RBF")
OBST_FILL, OBST_EDGE = "#E9E9E9", "#9A9A9A"

plt.rcParams.update({
    "font.family": "cmtt10", "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 14, "axes.linewidth": 1.1, "axes.edgecolor": "black",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linestyle": "--", "grid.linewidth": 0.9, "axes.axisbelow": True,
    "xtick.direction": "out", "ytick.direction": "out", "lines.linewidth": 2.3,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    "axes.spines.top": True, "axes.spines.right": True, "axes.titleweight": "normal",
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
})


def tint(c, a=0.72):
    r, g, b = colors.to_rgb(c)
    return (r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a)


def line(ax, x, y, c, ms=8, **kw):
    kw.setdefault("mfc", tint(c))
    return ax.plot(x, y, color=c, marker="o", ms=ms, mec=c, mew=1.7, **kw)[0]


def legend_below(fig, handles, labels, title, y=0.0, x=0.56):
    """One-row legend under the panels, with its title to the left of the entries."""
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(x, y),
                     handlelength=2.6, columnspacing=2.2)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x0 - 0.01, (bb.y0 + bb.y1) / 2, title, ha="right", va="center", fontsize=plt.rcParams["legend.fontsize"])
    return leg


def summary():
    return json.load(open(os.path.join(RES, "revised_summary.json")))


def main_rows(s, c):
    return sorted([r for r in s["main"] if r["controller"] == c], key=lambda r: r["k"])


def ci(r, key):
    return r[f"{key}_ci"] if r["n"] > 1 else (r[key], r[key])


def save(fig, name, rect=(0, 0.11, 1, 1), legend=None):
    fig.tight_layout(rect=rect)
    if legend:
        legend_below(fig, *legend)
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def fig_noise_sweep(s):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    hs = []
    for c in METHODS:
        g = main_rows(s, c)
        k = np.array([r["k"] for r in g])
        hs.append(line(axes[0], k, [100 * r["pos_err"] for r in g], COL[c]))
        line(axes[1], k, [100 * r["p_coll"] for r in g], COL[c])
        axes[1].fill_between(k, [100 * ci(r, "p_coll")[0] for r in g], [100 * ci(r, "p_coll")[1] for r in g],
                             color=COL[c], alpha=0.15, lw=0)
        line(axes[2], k, [r["safe_steps_mean"] for r in g], COL[c])
        axes[2].fill_between(k, [ci(r, "safe_steps_mean")[0] for r in g], [ci(r, "safe_steps_mean")[1] for r in g],
                             color=COL[c], alpha=0.15, lw=0)
    axes[0].set_title("Tracking Error (cm)")
    axes[1].set_title("Collision Rate (%)")
    axes[2].set_title("Steps Until First Collision")
    axes[1].set_ylim(-5, 105)
    axes[2].set_ylim(-5, 255)
    for ax in axes:
        ax.set_xlabel(r"Noise Scale ($\times\sigma$)")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
    save(fig, "noise_sweep.png", legend=(hs, [NAME[c] for c in METHODS], "Method"))


def fig_survival(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    hs = []
    t = np.arange(C.N_STEPS + 1)
    for ax, k in zip(axes, (0.5, 1.0)):
        for c in METHODS:
            rr = [r for r in rows if r["controller"] == c and r["noise_scale"] == k]
            first = np.array([r["first_coll_t"] if r["first_coll_t"] >= 0 else 10 ** 6 for r in rr])
            alive = 100 * (first[None, :] > t[:, None]).mean(1)
            h = line(ax, t, alive, COL[c], markevery=24)
            if k == 1.0:
                hs.append(h)
        for a, b in ((19, 25), (76, 82), (119, 125), (176, 182), (219, 225)):
            ax.axvspan(a, b, color="#EEEEEE", lw=0, zorder=0)
        ax.set_title(rf"Noise $\times {k:g}$")
        ax.set_xlabel("Time Step")
        ax.set_xlim(0, 240)
        ax.set_xticks([0, 60, 120, 180, 240])
        ax.set_ylim(-5, 105)
    axes[0].set_ylabel("Episodes Without a Collision (%)")
    save(fig, "survival.png", legend=(hs, [NAME[c] for c in METHODS], "Method"))


def fig_frontier(s):
    old = json.load(open(os.path.join(RES, "summary.json")))  # CEC margin runs (CEC is unaffected by GPI's risk model)
    cec = sorted([r for r in old["pilot_margin"] if r["k"] == 1.0], key=lambda r: r["pos_err"])
    gpi = sorted(s["lambda_sweep"], key=lambda r: r["pos_err"])
    rbf = [r for r in s["main"] if r["controller"] == "RBF" and r["k"] == 1.0]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    h1 = line(ax, [100 * r["pos_err"] for r in cec], [100 * r["p_coll"] for r in cec], DGRAY)
    h2 = line(ax, [100 * r["pos_err"] for r in gpi], [100 * r["p_coll"] for r in gpi], BLUE)
    h3 = line(ax, [100 * r["pos_err"] for r in rbf], [100 * r["p_coll"] for r in rbf], LGRAY, ls="none")
    for r in cec:
        m = float(r["controller"].replace("CEC-m", ""))
        off = {0.0: (-8, 10), 0.025: (8, -18), 0.05: (-10, 10), 0.1: (-40, -20)}[m]
        ax.annotate(f"m={m * 100:g}cm", (100 * r["pos_err"], 100 * r["p_coll"]), xytext=off,
                    textcoords="offset points", fontsize=12, color="#444444")
    for r in gpi:
        lam = r["controller"].replace("GPI-lam", "")
        off = {"10": (-18, -22), "100": (8, 6), "1000": (8, 6), "10000": (8, 6)}[lam]
        ax.annotate(r"$\lambda$=" + f"{int(lam):,}", (100 * r["pos_err"], 100 * r["p_coll"]), xytext=off,
                    textcoords="offset points", fontsize=12, color="#444444")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(7, 130)
    ax.set_ylim(-6, 110)
    ax.set_xlabel("Tracking Error (cm, log)")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title(r"Safety vs. Tracking at Noise $\times 1$")
    save(fig, "frontier.png", rect=(0, 0.1, 1, 1),
         legend=([h1, h2, h3], ["CEC + margin m", r"Tabular GPI, penalty $\lambda$", "RBF GPI"], "Method"))


def fig_value_rbf(s):
    import torch
    from gpi import GPI
    models = [("GPI", "revised_grid_medium_k1", "Tabular"), ("RBFavg", "revised_rbfavg_medium_k1", "RBF (averager)"),
              ("RBFls", "revised_rbf_medium_k1", "RBF (least squares)")]
    cols = {"GPI": BLUE, "RBFavg": LGRAY, "RBFls": LGRAY}
    styles = {"GPI": {}, "RBFavg": {}, "RBFls": dict(ls="--", mfc="white")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    xs = np.linspace(-1.2, 1.2, 97)
    t = 20
    hs, labs = [], []
    for key, name, lab in models:
        path = os.path.join(RES, "models", f"{name}.npz")
        if not os.path.exists(path):
            continue
        g = GPI.load(path, device="cpu")
        ev = g.V.evaluator()  # multilinear interpolation of node values (same as the Bellman backup)
        q = [torch.tensor(a.reshape(1, -1), dtype=torch.float32) for a in (xs, np.zeros_like(xs), np.zeros_like(xs))]
        with torch.no_grad():
            V = ev(torch.tensor([t]), *q).numpy().ravel()
        hs.append(line(axes[0], xs, V / 1000, cols[key], markevery=12, **styles[key]))
        labs.append(lab)
        hist = [h["V_mean"] for h in g.meta["history"]]
        line(axes[1], np.arange(1, len(hist) + 1), np.array(hist) / 1000, cols[key], markevery=2, **styles[key])
    p = np.stack([xs + C.REF[t, 0], np.full_like(xs, C.REF[t, 1])], 1)
    inside = C.clearance(p) < 0
    x0, x1 = xs[inside].min(), xs[inside].max()
    axes[0].axvspan(x0, x1, color="#EEEEEE", lw=0, zorder=0)
    axes[0].text((x0 + x1) / 2, 0.93, "inside C1", ha="center", va="center", fontsize=12, color="#555555",
                 transform=axes[0].get_xaxis_transform())
    axes[0].set_title("Value Across an Obstacle")
    axes[0].set_xlabel(r"Position Error $\tilde{e}_x$ (m)")
    axes[0].set_ylabel(r"$V$ ($\times 10^3$)")
    axes[1].set_title("Mean Value During GPI")
    axes[1].set_xlabel("GPI Iteration")
    axes[1].set_xticks([1, 5, 10, 15, 20])
    for ax in axes[:2]:
        ax.axhline(0, color="black", lw=1.0, ls=(0, (4, 3)))
    axes[1].text(0.97, 0.06, r"$V \geq 0$ must hold", ha="right", va="bottom", fontsize=12, color="#555555",
                 transform=axes[1].transAxes)
    for c, key in (("GPI", "GPI"), ("RBF", "RBFavg")):
        g = main_rows(s, c)
        line(axes[2], [r["k"] for r in g], [100 * r["p_coll"] for r in g], cols[key])
    axes[2].set_title("Collision Rate (%)")
    axes[2].set_xlabel(r"Noise Scale ($\times\sigma$)")
    axes[2].set_xticks([0, 0.5, 1, 1.5, 2])
    axes[2].set_ylim(-5, 105)
    save(fig, "value_rbf.png", legend=(hs, labs, "Value Function"))


def fig_phase(s):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    hs = []
    for i, c in enumerate(METHODS):
        g = sorted([r for r in s["phase_robustness"] if r["controller"] == c], key=lambda r: r["phase"])
        x = np.array([r["phase"] for r in g]) + (i - 1) * 1.6
        y = np.array([100 * r["p_hit"] for r in g])
        lo = y - np.array([100 * r["p_hit_ci"][0] for r in g])
        hi = np.array([100 * r["p_hit_ci"][1] for r in g]) - y
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=COL[c], elinewidth=1.4, capsize=4)
        hs.append(line(ax, x, y, COL[c]))
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Starting Phase of the Reference (step)")
    ax.set_ylim(-5, 108)
    ax.set_title(r"Collision Rate (%) by Starting Phase, Noise $\times 1$")
    save(fig, "phase.png", rect=(0, 0.12, 1, 1), legend=(hs, [NAME[c] for c in METHODS], "Method"))


def draw_env(ax, lim=3.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for (x, y, r) in C.OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, color=OBST_FILL, lw=0, zorder=1))
        ax.add_patch(plt.Circle((x, y), r + C.ROBOT_RADIUS, fill=False, ec=OBST_EDGE, lw=1.0, ls="--", zorder=1))
    r = C.REF
    ax.plot(np.r_[r[:, 0], r[0, 0]], np.r_[r[:, 1], r[0, 1]], color="black", lw=1.0, ls=":", zorder=2)


def fig_corridor(rows, seeds=range(20)):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, c in zip(axes, METHODS):
        draw_env(ax)
        for r in rows:
            if r["controller"] == c and r["noise_scale"] == 1.0 and r["seed"] in seeds:
                ax.plot(r["traj"][:, 0], r["traj"][:, 1], color=PATH_COL[c], lw=1.0, alpha=0.55, zorder=3)
        ax.set_xlim(0.2, 3.1)
        ax.set_ylim(-0.9, 2.1)
        ax.set_title(NAME[c])
        ax.text(2.35, 0.95, "C1", ha="center", va="center", fontsize=13, color="#666666", zorder=4)
        ax.text(1.0, 0.0, "C3", ha="center", va="center", fontsize=13, color="#666666", zorder=4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "corridor.png"), dpi=200)
    plt.close(fig)


def triangle(x, y, th, h=0.35, w=0.2):
    tri = np.array([[h, 0], [0, w / 2], [0, -w / 2]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ tri.T).T + [x, y]


def gif_rollout(rows, k=1.0, seed=0, stride=2):
    tr = {r["controller"]: r for r in rows if r["noise_scale"] == k and r["seed"] == seed and r["controller"] in COL}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=100)
    arts = []
    for ax, c in zip(axes, METHODS):
        draw_env(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(NAME[c], fontsize=15)
        col = PATH_COL[c]
        trail, = ax.plot([], [], color=col, lw=1.6, zorder=3)
        robot = plt.Polygon(triangle(0, 0, 0), color=col, zorder=5)
        refm = plt.Polygon(triangle(0, 0, 0), fc="white", ec="black", lw=1.0, zorder=4)
        ball = plt.Circle((0, 0), C.ROBOT_RADIUS, fill=False, ec=col, lw=1.0, zorder=5)
        for p in (refm, robot, ball):
            ax.add_patch(p)
        txt = ax.text(-3.05, -3.05, "", fontsize=11, va="bottom")
        arts.append((trail, robot, refm, ball, txt, tr[c], col))

    def update(i):
        for trail, robot, refm, ball, txt, r, col in arts:
            X = r["traj"]
            trail.set_data(X[: i + 1, 0], X[: i + 1, 1])
            robot.set_xy(triangle(*X[i]))
            refm.set_xy(triangle(*C.ref(i)))
            ball.center = (X[i, 0], X[i, 1])
            hit = r["clearance"][i] < 0
            ball.set_edgecolor("black" if hit else col)
            ball.set_linewidth(2.4 if hit else 1.0)
            txt.set_text(f"collisions: {int((r['clearance'][: i + 1] < 0).sum())}")
        return []

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=range(0, C.N_STEPS + 1, stride))
    anim.save(os.path.join(OUT, "rollout_k1.gif"), writer=animation.PillowWriter(fps=12))
    plt.close(fig)


def thumbnail(rows):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    draw_env(ax)
    for c in ("CEC", "GPI"):
        r = [r for r in rows if r["controller"] == c and r["noise_scale"] == 1.0 and r["seed"] == 0][0]
        ax.plot(r["traj"][:, 0], r["traj"][:, 1], color=COL[c], lw=1.8, label=NAME[c], zorder=3)
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-3.1, 3.1)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=14)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "thumb.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    s = summary()
    rows, _ = load("revised")  # results/rollouts_revised_part_k*.pkl
    fig_noise_sweep(s)
    fig_survival(rows)
    fig_frontier(s)
    fig_value_rbf(s)
    fig_phase(s)
    fig_corridor(rows)
    thumbnail(rows)
    gif_rollout(rows)
    print(sorted(os.listdir(OUT)))
