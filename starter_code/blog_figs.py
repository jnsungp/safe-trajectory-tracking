"""Figures for the blog post, in a minimal paper style (typewriter font, boxed axes, dashed grid,
blue for the method in focus and grays for the others).  python blog_figs.py -> results/figs/blog/"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors

import common as C
from analyze import RES, load, wilson

OUT = os.path.join(RES, "figs", "blog")
os.makedirs(OUT, exist_ok=True)

BLUE, DGRAY, LGRAY = "#598BE7", "#7F7F7F", "#B4B4B4"
COL = {"CEC": DGRAY, "GPI": BLUE, "RBF": LGRAY}
NAME = {"CEC": "CEC", "GPI": "Tabular GPI", "RBF": "RBF GPI"}
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


def line(ax, x, y, c, label=None, ms=8, **kw):
    return ax.plot(x, y, color=c, marker="o", ms=ms, mfc=tint(c), mec=c, mew=1.7, label=label, **kw)[0]


def legend_below(fig, handles, labels, title, ncol=None, y=0.0, x=0.56):
    """One-row legend under the panels, with its title to the left of the entries."""
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol or len(labels), bbox_to_anchor=(x, y),
                     handlelength=2.6, columnspacing=2.2)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x0 - 0.01, (bb.y0 + bb.y1) / 2, title, ha="right", va="center", fontsize=plt.rcParams["legend.fontsize"])
    return leg


def summary():
    return json.load(open(os.path.join(RES, "summary.json")))


# ------------------------------------------------------------------------------------------
def fig_noise_sweep():
    S = [r for r in summary()["main"] if r["controller"] in ("CEC", "GPI", "RBF")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    hs = {}
    for c in ("CEC", "GPI", "RBF"):
        g = sorted([r for r in S if r["controller"] == c], key=lambda r: r["k"])
        k = np.array([r["k"] for r in g])
        err = 100 * np.array([r["pos_err"] for r in g])
        pc = 100 * np.array([r["p_coll"] for r in g])
        lo = 100 * np.array([r["p_coll_ci"][0] if r["n"] > 1 else r["p_coll"] for r in g])
        hi = 100 * np.array([r["p_coll_ci"][1] if r["n"] > 1 else r["p_coll"] for r in g])
        clr = 100 * np.array([r["min_clear_med"] for r in g])
        hs[c] = line(axes[0], k, err, COL[c])
        line(axes[1], k, pc, COL[c])
        axes[1].fill_between(k, lo, hi, color=COL[c], alpha=0.15, lw=0)
        line(axes[2], k, clr, COL[c])
    axes[0].set_title("Tracking Error (cm)")
    axes[1].set_title("Collision Rate (%)")
    axes[2].set_title("Min Clearance (cm)")
    axes[1].set_ylim(-5, 105)
    axes[2].axhline(0, color="black", lw=1.0, ls=(0, (4, 3)))
    axes[2].text(1.98, -2, "collision", ha="right", va="top", fontsize=12, color="#555555")
    for ax in axes:
        ax.set_xlabel(r"Noise Scale ($\times\sigma$)")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    legend_below(fig, [hs[c] for c in hs], [NAME[c] for c in hs], "Method")
    fig.savefig(os.path.join(OUT, "noise_sweep.png"), dpi=200)
    plt.close(fig)


def fig_value_rbf():
    import torch
    from gpi import GPI
    models = [("GPI", "grid_medium_k1", "Tabular"), ("RBFavg", "rbfavg_medium_k1", "RBF (averager)"),
              ("RBFls", "rbf_medium_k1", "RBF (least squares)")]
    cols = {"GPI": BLUE, "RBFavg": LGRAY, "RBFls": LGRAY}
    styles = {"GPI": dict(), "RBFavg": dict(), "RBFls": dict(ls="--", mfc="white")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    xs = np.linspace(-1.2, 1.2, 97)
    t = 20
    hs = []
    for key, name, lab in models:
        g = GPI.load(os.path.join(RES, "models", f"{name}.npz"), device="cpu")
        ev = g.V.evaluator(exact=True) if g.config.value_type == "rbf" else g.V.evaluator()
        q = [torch.tensor(a.reshape(1, -1), dtype=torch.float32) for a in (xs, np.zeros_like(xs), np.zeros_like(xs))]
        with torch.no_grad():
            V = ev(torch.tensor([t]), *q).numpy().ravel()
        st = dict(mfc=tint(cols[key]))
        st.update(styles[key])
        hs.append(axes[0].plot(xs, V / 1000, color=cols[key], lw=2.3, marker="o", markevery=12, ms=8,
                               mec=cols[key], mew=1.7, **st)[0])
        hist = [h["V_mean"] for h in g.meta["history"]]
        axes[1].plot(np.arange(1, len(hist) + 1), np.array(hist) / 1000, color=cols[key], lw=2.3, marker="o",
                     markevery=2, ms=8, mec=cols[key], mew=1.7, **st)
    # obstacle interval along the slice
    p = np.stack([xs + C.REF[t, 0], np.full_like(xs, C.REF[t, 1])], 1)
    inside = C.clearance(p) < 0
    x0, x1 = xs[inside].min(), xs[inside].max()
    axes[0].axvspan(x0, x1, color="#EEEEEE", lw=0, zorder=0)
    axes[0].text((x0 + x1) / 2, 3.35, "inside C1", ha="center", va="center", fontsize=12, color="#555555")
    axes[0].set_title("Value Across an Obstacle")
    axes[0].set_xlabel(r"Position Error $\tilde{e}_x$ (m)")
    axes[0].set_ylabel(r"$V$ ($\times 10^3$)")
    axes[1].set_title("Mean Value During GPI")
    axes[1].set_xlabel("GPI Iteration")
    axes[1].set_xticks([1, 5, 10, 15, 20])
    for ax in axes[:2]:
        ax.axhline(0, color="black", lw=1.0, ls=(0, (4, 3)))
    axes[1].text(20, -0.12, r"$V \geq 0$ must hold", ha="right", va="top", fontsize=12, color="#555555")
    # collision rate vs noise: tabular vs averager (least squares omitted: it does not track at all)
    S = summary()["main"]
    for c, key in (("GPI", "GPI"), ("RBF", "RBFavg")):
        g = sorted([r for r in S if r["controller"] == c], key=lambda r: r["k"])
        line(axes[2], [r["k"] for r in g], [100 * r["p_coll"] for r in g], cols[key])
    axes[2].set_title("Collision Rate (%)")
    axes[2].set_xlabel(r"Noise Scale ($\times\sigma$)")
    axes[2].set_xticks([0, 0.5, 1, 1.5, 2])
    axes[2].set_ylim(-5, 105)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    legend_below(fig, hs, [m[2] for m in models], "Value Function")
    fig.savefig(os.path.join(OUT, "value_rbf.png"), dpi=200)
    plt.close(fig)


def fig_frontier():
    s = summary()
    cec = sorted([r for r in s["pilot_margin"] if r["k"] == 1.0], key=lambda r: r["stage_cost"])
    gpi = sorted(s["pilot_lambda"], key=lambda r: r["stage_cost"])
    rbf = [r for r in s["main"] if r["controller"] == "RBF" and r["k"] == 1.0]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    h1 = line(ax, [r["stage_cost"] for r in cec], [r["coll_steps"] for r in cec], DGRAY)
    h2 = line(ax, [r["stage_cost"] for r in gpi], [r["coll_steps"] for r in gpi], BLUE)
    h3 = line(ax, [r["stage_cost"] for r in rbf], [r["coll_steps"] for r in rbf], LGRAY, ls="none")
    for r in cec:
        m = r["controller"].replace("CEC-m", "")
        off = {"0": (8, 4), "0.025": (8, -16), "0.05": (-10, 10), "0.1": (-30, 10)}.get(m, (7, 5))
        ax.annotate(f"m={float(m) * 100:g}cm", (r["stage_cost"], r["coll_steps"]), xytext=off,
                    textcoords="offset points", fontsize=12, color="#444444")
    for r in gpi:
        lam = r["controller"].replace("GPI-lam", "")
        off = (7, 5) if lam != "100" else (7, -14)
        ax.annotate(r"$\lambda$=" + lam, (r["stage_cost"], r["coll_steps"]), xytext=off,
                    textcoords="offset points", fontsize=12, color="#444444")
    ax.set_xscale("log")
    ax.set_xlim(0.18, 45)
    ax.set_ylim(-1, 18)
    ax.set_xlabel("Mean Stage Cost (log)")
    ax.set_ylabel("Collision Steps per Episode")
    ax.set_title(r"Safety vs. Tracking at Noise $\times 1$")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    legend_below(fig, [h1, h2, h3], ["CEC + margin m", r"Tabular GPI, penalty $\lambda$", "RBF GPI"], "Method",
                 y=0.0, x=0.55)
    fig.savefig(os.path.join(OUT, "frontier.png"), dpi=200)
    plt.close(fig)


def fig_collision_timing():
    rows, _ = load("main")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bins = np.arange(0, 101, 2)
    hs = []
    for c in ("CEC", "GPI", "RBF"):
        ts = []
        for r in rows:
            if r["controller"] == c and r["noise_scale"] == 1.0:
                ts += list(np.where(r["clearance"] < 0)[0] % C.T_PERIOD)
        h, _ = np.histogram(ts, bins=bins)
        centers = bins[:-1] + 1
        hs.append(line(ax, centers, h / 200, COL[c], ms=6))
    for (a, b), lab in (((19, 25), "C1"), ((76, 82), "C2")):
        ax.axvspan(a, b, color="#EEEEEE", lw=0, zorder=0)
        ax.text((a + b) / 2, 2.85, f"ref. inside {lab}", ha="center", fontsize=12, color="#555555")
    ax.set_ylim(-0.1, 3.1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time within the Figure-Eight (steps)")
    ax.set_title(r"Collisions per Episode, per 2-step Bin (noise $\times 1$)")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    legend_below(fig, hs, [NAME[c] for c in ("CEC", "GPI", "RBF")], "Method", y=-0.01)
    fig.savefig(os.path.join(OUT, "collision_timing.png"), dpi=200)
    plt.close(fig)


def fig_compute():
    s = summary()
    tim = {t["controller"]: t for t in s["timing"]}
    off = [o for o in s["offline_timing"] if "RBF" not in o["grid"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    ax = axes[0]
    items = [("CEC N=10", "CEC", DGRAY), ("CEC N=20", "CEC N=20", DGRAY),
             ("GPI lookahead (medium)", "GPI lookahead", BLUE), ("GPI lookup (medium)", "GPI lookup", BLUE),
             ("RBF lookahead (exact kernel)", "RBF lookahead", LGRAY)]
    for i, (key, lab, c) in enumerate(items):
        v = tim[key]["ms_mean"]
        ax.barh(i, v, height=0.55, color=tint(c, 0.35), edgecolor=c, lw=1.6)
        ax.text(v * 1.15, i, f"{v:.2f} ms", va="center", fontsize=12)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[1] for it in items])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.08, 60)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_title("Online Time per Step (ms)")
    ns = np.array([o["n_states"] for o in off]) / 1e6
    line(axes[1], ns, [o["sec_per_iter"] for o in off], BLUE)
    for o, n in zip(off, ns):
        dx = -52 if o["grid"] == "fine" else 8
        axes[1].annotate(o["grid"], (n, o["sec_per_iter"]), xytext=(dx, 6), textcoords="offset points", fontsize=12)
    axes[1].set_ylim(-1, 33)
    axes[1].set_title("Offline Time per Iteration (s)")
    axes[1].set_xlabel("States (millions)")
    h1 = line(axes[2], ns, [o["peak_gpu_MB"] / 1024 for o in off], BLUE)
    h2 = line(axes[2], ns, [o["dense_P_table_GB"] for o in off], LGRAY, ls="--")
    axes[2].set_yscale("log")
    axes[2].set_title("Memory (GB, log)")
    axes[2].set_xlabel("States (millions)")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    legend_below(fig, [h1, h2], ["on-the-fly GPI (measured)", "stored transition table"], "Memory", y=-0.02)
    fig.savefig(os.path.join(OUT, "compute.png"), dpi=200)
    plt.close(fig)


def draw_env(ax, lim=3.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for (x, y, r) in C.OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, color=OBST_FILL, lw=0, zorder=1))
        ax.add_patch(plt.Circle((x, y), r + C.ROBOT_RADIUS, fill=False, ec=OBST_EDGE, lw=1.0, ls="--", zorder=1))
    r = C.REF
    ax.plot(np.r_[r[:, 0], r[0, 0]], np.r_[r[:, 1], r[0, 1]], color="black", lw=1.0, ls=":", zorder=2)


def fig_corridor(seeds=range(20)):
    rows, _ = load("main")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, c in zip(axes, ("CEC", "GPI", "RBF")):
        draw_env(ax)
        col = COL[c] if c != "RBF" else "#8C8C8C"
        for r in rows:
            if r["controller"] == c and r["noise_scale"] == 1.0 and r["seed"] in seeds:
                ax.plot(r["traj"][:, 0], r["traj"][:, 1], color=col, lw=1.0, alpha=0.55, zorder=3)
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


def gif_rollout(k=1.0, seed=0, stride=2):
    rows, _ = load("main")
    tr = {r["controller"]: r for r in rows if r["noise_scale"] == k and r["seed"] == seed and r["controller"] in COL}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=100)
    arts = []
    for ax, c in zip(axes, ("CEC", "GPI", "RBF")):
        draw_env(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(NAME[c], fontsize=15)
        col = COL[c] if c != "RBF" else "#8C8C8C"
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


def thumbnail():
    rows, _ = load("main")
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
    fig_noise_sweep()
    fig_value_rbf()
    fig_frontier()
    fig_collision_timing()
    fig_compute()
    fig_corridor()
    thumbnail()
    gif_rollout()
    print(sorted(os.listdir(OUT)))
