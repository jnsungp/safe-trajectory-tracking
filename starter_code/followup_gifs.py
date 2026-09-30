"""Animated figures for the follow-up sections of the blog post, in the style of blog_figs.py.

The CEC variants are re-run on seed 0 to record the plan they solve at every step; each re-run must reproduce the
stored rollout exactly. GPI episodes come straight from the stored rollouts.

Run after followup_analysis.py:  python followup_gifs.py -> results/figs/blog/*.gif
"""
import os
import pickle

import numpy as np

import common as C
from blog_figs import OUT, BLUE, DGRAY, OBST_EDGE, OBST_FILL, plt, triangle
from followup_analysis import arc_points
from followup_figs import DBLUE, ORANGE, obstacles, pick, reference, rows_of
from analyze import RES

PLANS = os.path.join(RES, "gif_plans.pkl")
SEED = 0
RUNS = {  # label -> (stored suite, stored controller label, noise scale)
    "RiskCEC-soft@0.25": ("riskcec", "RiskCEC-soft", 0.25),
    "RiskCEC-hard@0.25": ("riskcec", "RiskCEC-hard", 0.25),
    "RiskCEC-hard@1": ("riskcec", "RiskCEC-hard", 1.0),
    "RiskCEC-V@1": ("riskcec", "RiskCEC-V", 1.0),
    "RiskCEC-path-V@1": ("path", "RiskCEC-path-V", 1.0),
    "Hybrid@1": ("path", "Hybrid", 1.0),
}


def _spec(label, k):
    from experiments import _kname, hybrid_spec, riskcec_spec
    if label == "RiskCEC-soft":
        return riskcec_spec(k)
    if label == "RiskCEC-hard":
        return riskcec_spec(k, hard=True)
    if label == "RiskCEC-V":
        return riskcec_spec(k, hard=True, value_model=f"revised_grid_medium_{_kname(k)}")
    if label == "RiskCEC-path-V":
        return ("RiskCEC", {key: v for key, v in hybrid_spec(k)[1].items() if key != "policy_seed"})
    if label == "Hybrid":
        return hybrid_spec(k)
    raise KeyError(label)


def record_plans():
    """Re-run each CEC variant on seed 0 and keep, for every step, every solved plan (the warm-started one and,
    for the hybrid, the one started from GPI's plan) and which one was executed."""
    from experiments import make_controller
    out = {}
    for name, (suite, label, k) in RUNS.items():
        ctrl = make_controller(_spec(label, k))
        steps = []
        solve = ctrl._solve

        def recording_solve(t, x, z0, _solve=solve, _steps=steps):
            z, f, ok, it = _solve(t, x, z0)
            if not _steps or _steps[-1]["t"] != t:
                _steps.append(dict(t=t, cands=[]))
            X, _, _ = ctrl._unpack(z)
            _steps[-1]["cands"].append(dict(X=X[:2].T.copy(), f=f, ok=ok))
            return z, f, ok, it

        def step(t, x, ref, _steps=steps):
            u = ctrl(t, x, ref)
            X, _, _ = ctrl._unpack(ctrl.z_prev)  # the plan the controller executed and keeps as its warm start
            _steps[-1]["chosen"] = int(np.argmin([np.abs(c["X"] - X[:2].T).max() for c in _steps[-1]["cands"]]))
            return u

        ctrl._solve = recording_solve
        ctrl.reset()
        r = C.rollout(step, noise_scale=k, seed=SEED, keep_traj=True)
        ctrl._solve = solve
        stored = pick(rows_of(suite), label, k, [SEED])[0]
        diff = float(np.abs(r["traj"] - stored["traj"]).max())
        print(f"{name}: max |traj - stored| = {diff:.2e}", flush=True)
        assert diff < 1e-9, name
        out[name] = dict(traj=r["traj"], controls=r["controls"], clearance=r["clearance"], steps=steps)
    with open(PLANS, "wb") as fh:
        pickle.dump(out, fh)
    return out


def load_plans():
    if not os.path.exists(PLANS):
        return record_plans()
    with open(PLANS, "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------------------------------
def bare(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def robot_artists(ax, col):
    robot = plt.Polygon(triangle(0, 0, 0, h=0.22, w=0.13), color=col, zorder=7)
    ball = plt.Circle((0, 0), C.ROBOT_RADIUS, fill=False, ec=col, lw=1.0, zorder=7)
    refm = plt.Polygon(triangle(0, 0, 0, h=0.22, w=0.13), fc="white", ec="black", lw=1.0, zorder=6)
    for p in (refm, robot, ball):
        ax.add_patch(p)
    return robot, ball, refm


def set_robot(robot, ball, x, col, hit):
    robot.set_xy(triangle(x[0], x[1], x[2], h=0.22, w=0.13))
    ball.center = (x[0], x[1])
    ball.set_edgecolor("black" if hit else col)
    ball.set_linewidth(2.4 if hit else 1.0)


def save_gif(fig, update, frames, name, fps=12, colors=255):
    """Render every frame, quantize all of them to one shared palette without dithering, and let Pillow store
    only what changes between frames."""
    from PIL import Image
    imgs = []
    for fr in frames:
        update(fr)
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)
    probe = imgs[:: max(len(imgs) // 12, 1)]
    w, h = probe[0].size
    mosaic = Image.new("RGB", (w, h * len(probe)))
    for i, im in enumerate(probe):
        mosaic.paste(im, (0, h * i))
    pal = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    lut = np.array(pal.getpalette()[: 3 * colors]).reshape(-1, 3)
    lut[((lut - 255) ** 2).sum(1) < 3 * 6 ** 2] = 255  # Pillow maps white to any near-white entry; make them white
    pal.putpalette(lut.ravel().tolist())
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    path = os.path.join(OUT, name)
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {name}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


def risk_of(px, py, k):
    """One-sample collision probability of the inflated disks, as the risk-aware CEC scores a planned position."""
    from scipy.stats import ncx2
    s = max(0.04 * k, 0.01)
    p = np.zeros(np.shape(px))
    for (cx, cy, r) in C.OBSTACLES:
        d2 = ((np.asarray(px) - cx) ** 2 + (np.asarray(py) - cy) ** 2) / s ** 2
        p += ncx2.cdf(((r + C.ROBOT_RADIUS) / s) ** 2, 2, d2)
    return np.clip(p, 0, 1)


# ------------------------------------------------------------------------------------------
def gif_riskcec_c1(P, t0=8, t1=34):
    """Noise x0.25: the risk-aware CEC plans straight through C1, because its planned positions sit where the
    collision probability is flat; with the obstacle constraints kept, the plan bends around it."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=100)
    arts = []
    for ax, (name, title) in zip(axes, [("RiskCEC-soft@0.25", "Risk-aware CEC"),
                                        ("RiskCEC-hard@0.25", "+ Obstacle Constraints")]):
        bare(ax)
        obstacles(ax, (0, 2))
        reference(ax, 0, 60)
        ax.set_xlim(0.45, 3.25)
        ax.set_ylim(-1.0, 2.2)
        ax.set_title(title, fontsize=15)
        trail, = ax.plot([], [], color=ORANGE, lw=1.6, zorder=4)
        plan, = ax.plot([], [], color=ORANGE, lw=0.9, ls="--", zorder=5)
        dots = ax.scatter([], [], s=34, c=[], cmap="Greys", vmin=0, vmax=1, edgecolors=ORANGE, linewidths=1.0, zorder=6)
        robot, ball, refm = robot_artists(ax, ORANGE)
        txt = ax.text(0.52, -0.95, "", fontsize=11, va="bottom")
        arts.append((P[name], trail, plan, dots, robot, ball, refm, txt))
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=40)
    cb.set_label("collision probability the plan assigns to each planned position", fontsize=12)
    cb.ax.tick_params(labelsize=11)

    def update(t):
        for run, trail, plan, dots, robot, ball, refm, txt in arts:
            X, st = run["traj"], run["steps"][t]
            Xp = st["cands"][st["chosen"]]["X"]
            trail.set_data(X[: t + 1, 0], X[: t + 1, 1])
            plan.set_data(Xp[:, 0], Xp[:, 1])
            dots.set_offsets(Xp[1:])
            dots.set_array(risk_of(Xp[1:, 0], Xp[1:, 1], 0.25))
            set_robot(robot, ball, X[t], ORANGE, run["clearance"][t] < 0)
            refm.set_xy(triangle(*C.ref(t), h=0.22, w=0.13))
            txt.set_text(f"step {t}   collisions so far: {int((run['clearance'][: t + 1] < 0).sum())}")
        return []

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.2, wspace=0.04)
    save_gif(fig, update, list(range(t0, t1)) + [t1 - 1] * 8, "riskcec_c1.gif", fps=4)


def gif_riskcec_stall(P, gpi_row, stride=2):
    """Noise x1: the ten-step plan curls up at the minimum speed instead of paying for the gap, and the robot
    falls behind; GPI's value as the terminal cost removes the stall."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), dpi=100)
    runs = [("RiskCEC-hard@1", "+ Obstacle Constraints", ORANGE), ("RiskCEC-V@1", "+ GPI Value", ORANGE),
            (None, "Tabular GPI", BLUE)]
    arts = []
    for ax, (name, title, col) in zip(axes, runs):
        bare(ax)
        obstacles(ax, labels=False)
        reference(ax)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.6, 2.6)
        ax.set_title(title, fontsize=15)
        run = P[name] if name else dict(traj=gpi_row["traj"], clearance=gpi_row["clearance"], steps=None)
        trail, = ax.plot([], [], color=col, lw=1.4, zorder=4)
        plan, = ax.plot([], [], "o-", color=col, lw=0.9, ms=3.2, mfc="white", mew=0.9, zorder=5)
        robot, ball, refm = robot_artists(ax, col)
        txt = ax.text(-3.0, -2.55, "", fontsize=11, va="bottom")
        arts.append((run, trail, plan, robot, ball, refm, txt, col))

    def update(t):
        for run, trail, plan, robot, ball, refm, txt, col in arts:
            X = run["traj"]
            trail.set_data(X[max(t - 40, 0): t + 1, 0], X[max(t - 40, 0): t + 1, 1])
            if run["steps"] is not None and t < len(run["steps"]):
                st = run["steps"][t]
                Xp = st["cands"][st["chosen"]]["X"]
                plan.set_data(Xp[:, 0], Xp[:, 1])
            set_robot(robot, ball, X[t], col, run["clearance"][t] < 0)
            refm.set_xy(triangle(*C.ref(t), h=0.22, w=0.13))
            err = np.linalg.norm(X[t, :2] - C.ref(t)[:2])
            txt.set_text(f"error {err:4.2f} m   collisions {int((run['clearance'][: t + 1] < 0).sum())}")
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.03)
    save_gif(fig, update, range(0, C.N_STEPS, stride), "riskcec_stall.gif", fps=10)


def gif_gap_dash(rows, k=0.5, seeds=range(12), half=0.13):
    """Noise x0.5, tabular GPI with the sample risk: the first C2-C4 crossing of seeds 0-11, one after another.
    Left, the samples, which is all the collision check sees; right, the 5 cm gap magnified, with the path
    between the samples drawn black where it is inside an inflated obstacle."""
    from followup_analysis import gap_crossings
    mid = (C.OBSTACLES[1, :2] + C.OBSTACLES[3, :2]) / 2
    eps = []
    for sd in seeds:
        r = pick(rows, "GPI", k, [sd])[0]
        t = [c["t"] for c in gap_crossings(r["traj"], r["controls"]) if c["gap"] == "C2-C4"][0]
        fx, fy = arc_points(r["traj"][t - 1: t + 3], r["controls"][t - 1: t + 2], n_sub=401)
        eps.append(dict(X=r["traj"][t - 2: t + 4], fx=fx, fy=fy, fc=C.clearance(np.stack([fx, fy], -1)),
                        sclr=r["clearance"][t - 2: t + 4]))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), dpi=100)
    ax_s, ax_z = axes
    for ax in axes:
        bare(ax)
        obstacles(ax, (1, 3), labels=False)
    ax_s.set_xlim(-2.55, -0.8)
    ax_s.set_ylim(-1.35, 0.4)
    ax_s.add_patch(plt.Rectangle(mid - half, 2 * half, 2 * half, fill=False, ec="black", lw=0.9, zorder=6))
    ax_s.set_title("The Samples (All the Check Sees)", fontsize=14)
    ax_z.set_xlim(mid[0] - half, mid[0] + half)
    ax_z.set_ylim(mid[1] - half, mid[1] + half)
    ax_z.set_title("The Path Through the Gap, Magnified", fontsize=14)
    ax_z.annotate("", xy=mid + 0.0255 * np.array([0.818, 0.575]), xytext=mid - 0.0255 * np.array([0.818, 0.575]),
                  arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0, shrinkA=0, shrinkB=0), zorder=3)
    ax_z.text(mid[0] + 0.012, mid[1] - 0.03, "5 cm", fontsize=12, color="#444444", zorder=3)
    samp_old = ax_s.scatter([], [], s=22, color=BLUE, alpha=0.35, lw=0, zorder=4)
    samp_new = ax_s.scatter([], [], s=44, color=BLUE, edgecolors="white", linewidths=0.8, zorder=5)
    old_ok = [ax_z.plot([], [], color=BLUE, lw=1.0, alpha=0.35, zorder=4)[0] for _ in eps]
    old_bad = [ax_z.plot([], [], color="black", lw=2.4, alpha=0.6, zorder=5)[0] for _ in eps]
    cur_ok, = ax_z.plot([], [], color=BLUE, lw=2.0, zorder=6)
    cur_bad, = ax_z.plot([], [], color="black", lw=3.2, zorder=7)
    head, = ax_z.plot([], [], "o", color=BLUE, ms=7, mec="white", mew=1.0, zorder=8)
    txt_s = ax_s.text(0.02, 0.02, "", fontsize=11, va="bottom", transform=ax_s.transAxes)
    txt_z = ax_z.text(0.02, 0.02, "", fontsize=11, va="bottom", transform=ax_z.transAxes,
                      bbox=dict(fc="white", ec="none", pad=1.5), zorder=9)
    n_sub = 18  # frames per step
    frames = [(i, j) for i in range(len(eps)) for j in range(3 * n_sub + 1)] + [(len(eps) - 1, 3 * n_sub)] * 12

    def masked(x, y, bad):
        return np.where(bad, x, np.nan), np.where(bad, y, np.nan)

    def update(fr):
        i, j = fr
        e = eps[i]
        fx, fy, fc = e["fx"].ravel(), e["fy"].ravel(), e["fc"].ravel()
        n = int(round(j / (3 * n_sub) * (len(fx) - 1)))
        step = min(j // n_sub, 2)
        shown = np.concatenate([eps[m]["X"] for m in range(i)] + [e["X"][: step + 2]])
        samp_old.set_offsets(shown[: -1] if len(shown) > 1 else np.empty((0, 2)))
        samp_new.set_offsets(e["X"][step + 1: step + 2])
        for m in range(len(eps)):
            if m < i:
                x, y, c = eps[m]["fx"].ravel(), eps[m]["fy"].ravel(), eps[m]["fc"].ravel()
                old_ok[m].set_data(x, y)
                old_bad[m].set_data(*masked(x, y, c < 0))
            else:
                old_ok[m].set_data([], [])
                old_bad[m].set_data([], [])
        cur_ok.set_data(fx[: n + 1], fy[: n + 1])
        cur_bad.set_data(*masked(fx[: n + 1], fy[: n + 1], fc[: n + 1] < 0))
        head.set_data(fx[n: n + 1], fy[n: n + 1])
        n_samp_bad = sum(int((eps[m]["sclr"] < 0).sum()) for m in range(i + 1))
        done = [eps[m]["fc"].min() < 0 for m in range(i)] + [fc[: n + 1].min() < 0]
        txt_s.set_text(f"seed {i}   samples inside an obstacle: {n_samp_bad}")
        txt_z.set_text(f"crossings whose path touches an obstacle: {sum(done)} of {i + 1}")
        return []

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)
    save_gif(fig, update, frames, "gap_dash.gif", fps=18)


def gif_hybrid_route(P, gpi_path_rows, bg_rows, t0=44, t1=100):
    """Noise x1, the left-hand lobe. The warm-started plan follows the reference down the outer side of C2 into
    the dead end at the C2-C4 gap; the hybrid also solves from GPI's plan and keeps whichever is cheaper."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.9), dpi=100)
    gp = pick(gpi_path_rows, "GPI-path", 1.0, [SEED])[0]
    runs = [("RiskCEC-path-V@1", "Risk-aware CEC + GPI Value", ORANGE),
            ("Hybrid@1", "+ GPI Plan as a Start", ORANGE),
            (None, "Tabular GPI, Path Risk", DBLUE)]
    arts = []
    for ax, (name, title, col) in zip(axes, runs):
        bare(ax)
        obstacles(ax, (1, 3), labels=False)
        reference(ax, 40, 100)
        key = {"RiskCEC-path-V@1": "RiskCEC-path-V", "Hybrid@1": "Hybrid", None: "GPI-path"}[name]
        for r in pick(bg_rows, key, 1.0, range(1, 20)):  # nineteen other seeds, faint
            ax.plot(r["traj"][t0:t1 + 1, 0], r["traj"][t0:t1 + 1, 1], color=col, lw=0.8, alpha=0.18, zorder=2)
        ax.set_xlim(-2.8, 0.4)
        ax.set_ylim(-2.3, 2.3)
        ax.set_title(title, fontsize=14)
        run = P[name] if name else dict(traj=gp["traj"], controls=gp["controls"], clearance=gp["clearance"],
                                          steps=None)
        trail, = ax.plot([], [], color=col, lw=1.8, zorder=4)
        chosen, = ax.plot([], [], "o-", color=col, lw=1.0, ms=3.4, mfc="white", mew=0.9, zorder=6)
        other, = ax.plot([], [], "o--", color="#999999", lw=0.9, ms=3.0, mfc="white", mew=0.8, zorder=5)
        robot, ball, refm = robot_artists(ax, col)
        txt = ax.text(-2.72, -2.25, "", fontsize=11, va="bottom")
        _, py_ = arc_points(run["traj"], run["controls"])
        px_, _ = arc_points(run["traj"], run["controls"])
        pclr = C.clearance(np.stack([px_, py_], -1)).min(axis=1)
        arts.append((run, trail, chosen, other, robot, ball, refm, txt, col, pclr))

    def update(t):
        for run, trail, chosen, other, robot, ball, refm, txt, col, pclr in arts:
            X = run["traj"]
            trail.set_data(X[t0: t + 1, 0], X[t0: t + 1, 1])
            if run["steps"] is not None:
                st = run["steps"][t]
                c = st["chosen"]
                Xc = st["cands"][c]["X"]
                chosen.set_data(Xc[:, 0], Xc[:, 1])
                if len(st["cands"]) > 1:
                    Xo = st["cands"][1 - c]["X"]
                    other.set_data(Xo[:, 0], Xo[:, 1])
            hit = pclr[t - 1] < 0 if t > 0 else False
            set_robot(robot, ball, X[t], col, hit)
            refm.set_xy(triangle(*C.ref(t), h=0.22, w=0.13))
            txt.set_text(f"steps with a collision along the path: {int((pclr[t0:t] < 0).sum())}")
        return []

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03)
    save_gif(fig, update, list(range(t0, t1)) + [t1 - 1] * 8, "hybrid_route.gif", fps=6)


if __name__ == "__main__":
    P = load_plans()
    gif_riskcec_c1(P)
    gif_riskcec_stall(P, pick(rows_of("revised_part_k1"), "GPI", 1.0, [SEED])[0])
    gif_gap_dash(rows_of("revised_part_k0p5"))
    path_rows = rows_of("path")
    gif_hybrid_route(P, path_rows, path_rows)
