"""Figures for the results-first revised blog post. Run after revised_analysis.py."""
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze as A
import common as C


OUT = os.path.join(A.RES, "figs", "blog")
os.makedirs(OUT, exist_ok=True)
SITE_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "blog_build", "blog",
                                   "safe-trajectory-tracking", "assets"))
os.makedirs(SITE_ASSETS, exist_ok=True)
COLORS = {"CEC": "#747474", "GPI": "#3979D6", "RBF": "#C67A36"}
LABELS = {"CEC": "CEC", "GPI": "Tabular GPI", "RBF": "RBF GPI"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.2, "savefig.facecolor": "white"})


def data():
    with open(os.path.join(A.RES, "revised_summary.json")) as fh:
        return json.load(fh)


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180)
    shutil.copy2(path, os.path.join(SITE_ASSETS, name))
    plt.close(fig)


def noise_sweep(s):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for c in COLORS:
        rows = sorted((r for r in s["main"] if r["controller"] == c), key=lambda r: r["k"])
        k = [r["k"] for r in rows]
        color = COLORS[c]
        axes[0].plot(k, [r["pos_err"] for r in rows], "o-", color=color, label=LABELS[c])
        axes[1].plot(k, [100 * r["p_coll"] for r in rows], "o-", color=color)
        axes[2].plot(k, [r["safe_steps_mean"] for r in rows], "o-", color=color)
        noisy = [r for r in rows if r["n"] > 1]
        axes[1].fill_between([r["k"] for r in noisy],
                             [100 * r["p_coll_ci"][0] for r in noisy],
                             [100 * r["p_coll_ci"][1] for r in noisy], color=color, alpha=0.1)
    axes[0].set_ylabel("Mean position error (m)")
    axes[1].set_ylabel("Episodes with a collision (%)")
    axes[2].set_ylabel("Mean steps until first collision\n(capped at 240)")
    axes[1].set_ylim(-4, 104)
    axes[2].set_ylim(0, 245)
    for ax in axes:
        ax.set_xlabel("Noise scale (×σ)")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "revised_noise_sweep.png")


def survival(rows):
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for c in COLORS:
        selected = [r for r in rows if r["controller"] == c and r["noise_scale"] == 1.0]
        first = np.array([r["first_coll_t"] for r in selected])
        first = np.where(first < 0, C.N_STEPS + 1, first)
        t = np.arange(C.N_STEPS + 1)
        p = (first[:, None] > t[None, :]).mean(0)
        ax.step(t, p, where="post", color=COLORS[c], label=LABELS[c], lw=2)
    ax.set(xlabel="Step", ylabel="Fraction with no sampled collision", xlim=(0, C.N_STEPS), ylim=(0, 1.04))
    ax.legend(frameon=False)
    save(fig, "revised_survival_k1.png")


def corridor(rows):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    for ax, c in zip(axes, COLORS):
        for x, y, r in C.OBSTACLES:
            ax.add_patch(plt.Circle((x, y), r + C.ROBOT_RADIUS, facecolor="#F2F2F2",
                                    edgecolor="#999999", lw=1))
        ax.plot(C.REF[:, 0], C.REF[:, 1], color="#555555", lw=1, ls=":", alpha=0.7)
        for rec in rows:
            if rec["controller"] == c and rec["noise_scale"] == 1.0 and rec["seed"] < 20:
                path = rec["traj"]
                ax.plot(path[:, 0], path[:, 1], color=COLORS[c], lw=0.7, alpha=0.55)
        ax.set(xlim=(0.2, 3.15), ylim=(-0.9, 2.1), title=LABELS[c], xlabel="x (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (m)")
    save(fig, "revised_corridor_k1.png")


def risk_calibration(s):
    rows = s["corridor_risk"]
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    k = np.array([r["k"] for r in rows])
    ax.plot(k, [100 * r["exact_disk_union"] for r in rows], "o-", color=COLORS["GPI"], label="Exact disk union")
    ax.plot(k, [100 * r["old_independent_halfplane"] for r in rows], "s--", color=COLORS["CEC"],
            label="Earlier risk estimate")
    ax.set(xlabel="Noise scale (×σ)", ylabel="One-step collision probability (%)",
           title="At the center of the 5 cm corridor", ylim=(0, 80))
    ax.legend(frameon=False)
    save(fig, "revised_risk_calibration.png")


def phase_sweep(s):
    if "phase_robustness" not in s:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for c in COLORS:
        rr = sorted((r for r in s["phase_robustness"] if r["controller"] == c),
                    key=lambda r: r["phase"])
        ax.plot([r["phase"] for r in rr], [100 * r["p_hit"] for r in rr], "o-",
                color=COLORS[c], label=LABELS[c])
    ax.set(xlabel="Starting phase (reference steps)", ylabel="Episodes with a collision (%)",
           title="Two complete periods, aligned initial state", ylim=(0, 105))
    ax.set_xticks([0, 25, 50, 75])
    ax.legend(frameon=False)
    save(fig, "revised_phase_sweep.png")


def penalty_sweep(s):
    if "lambda_sweep" not in s:
        return
    rows = sorted(s["lambda_sweep"], key=lambda r: int(r["controller"].split("lam")[-1]))
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    x = [r["pos_err"] for r in rows]
    y = [100 * r["p_coll"] for r in rows]
    ax.plot(x, y, "o-", color=COLORS["GPI"])
    for r in rows:
        penalty = int(r["controller"].split("lam")[-1])
        offset = (5, -18) if penalty == 100 else (5, 7)
        ax.annotate("λ=" + r["controller"].split("lam")[-1],
                    (r["pos_err"], 100 * r["p_coll"]), xytext=offset,
                    textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean position error (m)", ylabel="Episodes with a collision (%)",
           title="Risk penalty sweep at noise ×1", ylim=(-4, 104))
    save(fig, "revised_penalty_sweep.png")


if __name__ == "__main__":
    s = data()
    rows, _ = A.load("revised")
    noise_sweep(s)
    survival(rows)
    corridor(rows)
    risk_calibration(s)
    phase_sweep(s)
    penalty_sweep(s)
