"""Animated side-by-side rollouts (GIF) for the write-up.  python make_media.py"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import common as C
from analyze import AXIS, CRIT, FIG, INK2, MUTED, RES, STYLE, SURF, draw_env


def triangle(x, y, th, h=0.35, w=0.2):
    tri = np.array([[h, 0], [0, w / 2], [0, -w / 2]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ tri.T).T + [x, y]


def make_gif(rows, k=1.0, seed=0, ctrls=("CEC", "GPI", "RBF"), stride=2, out="rollout_k1.gif"):
    trajs = {}
    for r in rows:
        if r["noise_scale"] == k and r["seed"] == seed and r["controller"] in ctrls:
            trajs[r["controller"]] = r
    ctrls = [c for c in ctrls if c in trajs]
    fig, axes = plt.subplots(1, len(ctrls), figsize=(3.1 * len(ctrls), 3.3), dpi=110)
    arts = []
    for ax, c in zip(axes, ctrls):
        draw_env(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(c, color=INK2)
        trail, = ax.plot([], [], color=STYLE[c]["color"], lw=1.5)
        robot = plt.Polygon(triangle(0, 0, 0), color=STYLE[c]["color"])
        refm = plt.Polygon(triangle(0, 0, 0), color=MUTED, alpha=0.8)
        ball = plt.Circle((0, 0), C.ROBOT_RADIUS, fill=False, ec=STYLE[c]["color"], lw=1)
        ax.add_patch(refm)
        ax.add_patch(robot)
        ax.add_patch(ball)
        txt = ax.text(-3.0, -3.1, "", fontsize=7, color=INK2, va="top")
        arts.append((trail, robot, refm, ball, txt, trajs[c]))
    frames = list(range(0, C.N_STEPS + 1, stride))

    def update(i):
        out = []
        for trail, robot, refm, ball, txt, r in arts:
            X = r["traj"]
            trail.set_data(X[: i + 1, 0], X[: i + 1, 1])
            robot.set_xy(triangle(*X[i]))
            rr = C.ref(i)
            refm.set_xy(triangle(*rr))
            ball.center = (X[i, 0], X[i, 1])
            hit = r["clearance"][i] < 0
            ball.set_edgecolor(CRIT if hit else AXIS)
            ball.set_linewidth(2.0 if hit else 1.0)
            ncol = int((r["clearance"][: i + 1] < 0).sum())
            txt.set_text(f"t={i * C.DT:5.1f}s  |e|={np.linalg.norm(X[i, :2] - rr[:2]):.2f} m  collisions={ncol}")
            out += [trail, robot, refm, ball, txt]
        return out

    fig.suptitle(f"Same noise realization (σ×{k:g}, seed {seed}); gray triangle = reference, "
                 "ring turns red in collision", fontsize=8, color=INK2)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path = os.path.join(FIG, out)
    anim.save(path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print(path, os.path.getsize(path) / 1e6, "MB")


if __name__ == "__main__":
    from analyze import load
    rows, _ = load("main")
    make_gif(rows, k=1.0, seed=0, out="rollout_k1.gif")
    make_gif(rows, k=0.0, seed=0, out="rollout_k0.gif")
