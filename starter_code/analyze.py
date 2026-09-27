"""Statistics and figures for the hypothesis tests.  python analyze.py  ->  results/summary.json, results/figs/*.png"""
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as C

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)

# reference palette (dataviz skill), fixed order; markers are the secondary encoding
STYLE = {
    "CEC": dict(color="#2a78d6", marker="o"),
    "GPI": dict(color="#eb6834", marker="s"),
    "RBF": dict(color="#1baf7a", marker="^"),
    "GPI-detmodel": dict(color="#eda100", marker="D"),
    "RBF-LS": dict(color="#e87ba4", marker="v"),
    "GPI-lookup": dict(color="#008300", marker="P"),
}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT = "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

CORRIDORS = [(0, 2), (1, 3)]  # (C1, C3) and (C2, C4): 5 cm gap between the inflated obstacles


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def _seg_cross(p, q, a, b):
    """Vectorized: does segment p->q (arrays (n,2)) cross segment a->b?"""
    def orient(u, v, w):
        return np.sign((v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0]))
    a = np.broadcast_to(a, p.shape)
    b = np.broadcast_to(b, p.shape)
    return (orient(p, q, a) != orient(p, q, b)) & (orient(a, b, p) != orient(a, b, q))


def corridor_crossings(traj):
    p, q = traj[:-1, :2], traj[1:, :2]
    n = 0
    for i, j in CORRIDORS:
        n += int(_seg_cross(p, q, C.OBSTACLES[i, :2], C.OBSTACLES[j, :2]).sum())
    return n


def seg_clearance(traj):
    """Clearance along the straight chords x_t -> x_{t+1} (over-estimates penetration on curved paths)."""
    P, Q = traj[:-1, :2], traj[1:, :2]
    d = Q - P
    dd = np.maximum((d ** 2).sum(1), 1e-12)
    best = np.inf
    for (cx, cy, r) in C.OBSTACLES:
        c = np.array([cx, cy])
        tt = np.clip(((c - P) * d).sum(1) / dd, 0, 1)
        dist = np.linalg.norm(c - (P + tt[:, None] * d), axis=1) - (r + C.ROBOT_RADIUS)
        best = min(best, dist.min())
    return best


def arc_clearance(traj, controls, n_sub=21):
    """Clearance along the continuous path implied by eq. (1): the constant-(v, w) unicycle arc from x_t,
    plus the step's noise w_t = x_{t+1} - f(x_t, u_t) blended in linearly over the step. For noise x0 this is
    the exact continuous trajectory. (The problem constrains only sampled states.)"""
    X, U = traj[:-1], controls
    Wn = traj[1:] - C.f(X, U)
    Wn[:, 2] = C.wrap(Wn[:, 2])
    s = np.linspace(0, C.DT, n_sub)[None, :]  # (1, n_sub)
    v, om, th = U[:, :1], U[:, 1:2], X[:, 2:3]
    half = om * s / 2
    step = s * np.sinc(half / np.pi) * v
    px = X[:, :1] + step * np.cos(th + half) + (s / C.DT) * Wn[:, :1]
    py = X[:, 1:2] + step * np.sin(th + half) + (s / C.DT) * Wn[:, 1:2]
    return float(C.clearance(np.stack([px, py], -1)).min())


def n_passages(n_steps=C.N_STEPS):
    """Number of times the reference itself passes a corridor (reference crosses the C1-C3 / C2-C4 segment)."""
    r = C.ref(np.arange(n_steps + 1))
    return corridor_crossings(r)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean()))
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def mcnemar(a, b):
    """Exact McNemar test on paired booleans."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue), n01, n10


def paired(df, ca, cb, k, col):
    A = df[(df.controller == ca) & (df.noise_scale == k)].set_index("seed")[col]
    B = df[(df.controller == cb) & (df.noise_scale == k)].set_index("seed")[col]
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx].values, B.loc[idx].values
    return A, B


def load(suite):
    """Loads results/rollouts_<suite>.pkl and any rollouts_<suite>_part*.pkl (deduplicated)."""
    import glob
    paths = sorted(glob.glob(os.path.join(RES, f"rollouts_{suite}.pkl")) +
                   glob.glob(os.path.join(RES, f"rollouts_{suite}_part*.pkl")))
    if not paths:
        return None, None
    rows, seen = [], set()
    for path in paths:
        with open(path, "rb") as fh:
            for r in pickle.load(fh):
                key = (r["controller"], r["noise_scale"], r["seed"])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    for r in rows:
        r["min_clear_chord"] = seg_clearance(r["traj"])
        r["min_clear_seg"] = arc_clearance(r["traj"], r["controls"])
    for r in rows:
        r["corridor_crossings"] = corridor_crossings(r["traj"])
    scal = [{k: v for k, v in r.items() if np.isscalar(v) or isinstance(v, (bool, str))} for r in rows]
    return rows, pd.DataFrame(scal)


def summarize(df):
    out = []
    npass = n_passages()
    for (c, k), g in df.groupby(["controller", "noise_scale"]):
        n = len(g)
        kc = int(g.collided.sum())
        lo, hi = wilson(kc, n)
        row = dict(controller=c, k=k, n=n,
                   pos_err=g.mean_pos_err.mean(), pos_err_ci=boot_ci(g.mean_pos_err),
                   th_err=g.mean_abs_th_err.mean(),
                   stage_cost=g.mean_stage_cost.mean(), stage_cost_ci=boot_ci(g.mean_stage_cost),
                   disc_cost=g.disc_cost.mean(), disc_cost_ci=boot_ci(g.disc_cost),
                   p_coll=kc / n, p_coll_ci=(lo, hi), coll_steps=g.n_coll_steps.mean(),
                   p_coll_seg=(g.min_clear_seg < 0).mean(), min_clear_seg_med=g.min_clear_seg.median(),
                   p_coll_chord=(g.min_clear_chord < 0).mean(),
                   p_pen2cm=(g.min_clearance < -0.02).mean(), p_pen5cm=(g.min_clearance < -0.05).mean(),
                   min_clear_med=g.min_clearance.median(), min_clear_p05=g.min_clearance.quantile(0.05),
                   corridor_frac=g.corridor_crossings.mean() / npass,
                   ctrl_ms=g.ctrl_ms_mean.mean(), oob=g.oob.mean())
        for extra in ("solver_fail_rate", "plan_slack_rate"):
            if extra in g:
                row[extra] = g[extra].mean()
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------
def draw_env(ax, lim=3.2):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for (x, y, r) in C.OBSTACLES:
        ax.add_patch(plt.Circle((x, y), r, color=AXIS, alpha=0.9, lw=0))
        ax.add_patch(plt.Circle((x, y), r + C.ROBOT_RADIUS, fill=False, ec=MUTED, lw=0.8))
    ax.add_patch(plt.Rectangle((-3, -3), 6, 6, fill=False, ec=MUTED, lw=0.8))
    r = C.REF
    ax.plot(np.r_[r[:, 0], r[0, 0]], np.r_[r[:, 1], r[0, 1]], color=INK2, lw=1, alpha=0.6)


def fig_trajectories(rows, ks=(0.0, 1.0), ctrls=("CEC", "GPI", "RBF"), seed=0):
    fig, axes = plt.subplots(len(ks), len(ctrls), figsize=(3.2 * len(ctrls), 3.3 * len(ks)))
    for i, k in enumerate(ks):
        for j, c in enumerate(ctrls):
            ax = axes[i, j]
            draw_env(ax)
            rr = [r for r in rows if r["controller"] == c and r["noise_scale"] == k and r["seed"] == seed]
            if rr:
                X = rr[0]["traj"]
                ax.plot(X[:, 0], X[:, 1], color=STYLE[c]["color"], lw=1.5)
                bad = rr[0]["clearance"] < 0
                ax.plot(X[bad, 0], X[bad, 1], "x", color=CRIT, ms=6, mew=1.5)
                ax.set_title(f"{c}  σ×{k:g}  |  err {rr[0]['mean_pos_err']:.3f} m, "
                             f"{int(bad.sum())} coll. steps", fontsize=8)
            ax.set_xticks([-3, 0, 3])
            ax.set_yticks([-3, 0, 3])
    fig.suptitle("Closed-loop paths (seed 0). Gray: obstacles; thin ring: robot-inflated boundary; "
                 "red ×: states in collision", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_trajectories.png"), dpi=150)
    plt.close(fig)


def fig_corridor_zoom(rows, k=1.0, seeds=range(20), ctrls=("CEC", "GPI", "RBF")):
    fig, axes = plt.subplots(1, len(ctrls), figsize=(3.3 * len(ctrls), 3.3))
    for ax, c in zip(axes, ctrls):
        draw_env(ax)
        for r in rows:
            if r["controller"] == c and r["noise_scale"] == k and r["seed"] in seeds:
                X = r["traj"]
                ax.plot(X[:, 0], X[:, 1], color=STYLE[c]["color"], lw=0.8, alpha=0.5)
        ax.set_xlim(0.2, 3.1)
        ax.set_ylim(-0.9, 2.1)
        ax.set_title(f"{c}, σ×{k:g}, 20 seeds")
    fig.suptitle("Right lobe: the C1–C3 corridor is 5 cm wide after inflating by the robot radius",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_corridor_zoom.png"), dpi=150)
    plt.close(fig)


def fig_noise_sweep(S, ctrls=("CEC", "GPI", "RBF", "GPI-detmodel")):
    panels = [("pos_err", "pos_err_ci", "mean position error [m]"),
              ("stage_cost", "stage_cost_ci", "mean stage cost ℓ"),
              ("p_coll", "p_coll_ci", "P(episode has a collision)"),
              ("min_clear_med", None, "median min clearance [m]")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, (col, ci, lab) in zip(axes, panels):
        for c in ctrls:
            g = S[S.controller == c].sort_values("k")
            if g.empty:
                continue
            ax.plot(g.k, g[col], color=STYLE[c]["color"], marker=STYLE[c]["marker"], ms=5, label=c,
                    mec=SURF, mew=1)
            if ci:
                gg = g[g.n > 1]  # k = 0 is a single deterministic run: no interval
                lo = np.array([v[0] for v in gg[ci]])
                hi = np.array([v[1] for v in gg[ci]])
                ax.fill_between(gg.k, lo, hi, color=STYLE[c]["color"], alpha=0.12, lw=0)
        ax.set_xlabel("noise scale  (σ × k)")
        ax.set_title(lab)
        if col == "stage_cost":
            ax.set_yscale("log")
        if col == "min_clear_med":
            ax.axhline(0, color=CRIT, lw=1)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_noise_sweep.png"), dpi=150)
    plt.close(fig)


def fig_tradeoff(S, ctrls=("CEC", "GPI", "RBF", "GPI-detmodel")):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for c in ctrls:
        g = S[(S.controller == c) & (S.k > 0)].sort_values("k")
        ax.plot(g.stage_cost, g.p_coll, color=STYLE[c]["color"], marker=STYLE[c]["marker"], ms=6, lw=1,
                label=c, mec=SURF, mew=1)
        for j, (_, r) in enumerate(g.iterrows()):
            if c == "CEC" and 0 < j < len(g) - 1:
                continue  # CEC points overlap at P = 1; label the ends only
            ax.annotate(f"k={r.k:g}", (r.stage_cost, r.p_coll), fontsize=7, color=INK2,
                        xytext=(4, 3 if c != "CEC" else -11), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean stage cost ℓ (tracking + effort, log)")
    ax.set_ylabel("P(episode has a collision)")
    ax.set_title("Tracking-vs-safety trade-off (labels = noise scale k)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_tradeoff.png"), dpi=150)
    plt.close(fig)


def fig_collision_timing(rows, k=1.0, ctrls=("CEC", "GPI", "RBF")):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    bins = np.arange(0, 101, 2)
    for c in ctrls:
        ts = []
        for r in rows:
            if r["controller"] == c and r["noise_scale"] == k:
                bad = np.where(r["clearance"] < 0)[0]
                ts += list(bad % C.T_PERIOD)
        if ts:
            h, _ = np.histogram(ts, bins=bins)
            ax.step(bins[:-1], h, where="post", color=STYLE[c]["color"], label=c)
    for a, b in ((19, 25), (76, 82)):
        ax.axvspan(a, b, color=GRID, alpha=0.6, lw=0)
    ax.set_xlabel("time within reference period (t mod 100);  shaded: reference inside inflated C1 / C2")
    ax.set_ylabel("# states in collision")
    ax.set_title(f"Where collisions happen (σ×{k:g}, all seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_collision_timing.png"), dpi=150)
    plt.close(fig)


def fig_value_slices(t=20, models=(("GPI (tabular)", "grid_medium_k1"), ("RBF averager", "rbfavg_medium_k1"),
                                   ("RBF least-squares", "rbf_medium_k1"))):
    import torch
    from gpi import GPI
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 3.4))
    prof = {}
    for ax, (lab, name) in zip(axes, models):
        path = os.path.join(RES, "models", f"{name}.npz")
        if not os.path.exists(path):
            continue
        g = GPI.load(path, device="cpu")
        n = 121
        xs = np.linspace(-1.2, 1.2, n)
        EX, EY = np.meshgrid(xs, xs, indexing="ij")
        ev = g.V.evaluator(exact=True) if g.config.value_type == "rbf" else g.V.evaluator()
        q = [torch.tensor(a.reshape(1, -1), dtype=torch.float32) for a in (EX, EY, np.zeros_like(EX))]
        with torch.no_grad():
            V = ev(torch.tensor([t]), *q).numpy().reshape(n, n)
        im = ax.pcolormesh(xs, xs, np.log10(np.clip(V, 1e-1, None)).T, cmap="Blues", shading="auto",
                           vmin=0, vmax=4.5)
        if (V < 0).any():
            ax.contour(xs, xs, V.T, levels=[0], colors=[INK], linewidths=0.8)
        for (x, y, r) in C.OBSTACLES:
            cx, cy = x - C.REF[t, 0], y - C.REF[t, 1]
            ax.add_patch(plt.Circle((cx, cy), r + C.ROBOT_RADIUS, fill=False, ec=CRIT, lw=1))
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(f"{lab}: log10 V(t={t}, ẽx, ẽy, θ̃=0)")
        ax.set_xlabel("ẽx [m]")
        ax.set_ylabel("ẽy [m]")
        fig.colorbar(im, ax=ax, fraction=0.046)
        prof[lab] = (xs, V[:, n // 2], V.min())
    ax = axes[-1]
    for (lab, (xs, v, vmin)), col in zip(prof.items(), ("#eb6834", "#1baf7a", "#e87ba4")):
        ax.plot(xs, v, color=col, label=f"{lab} (min {vmin:.0f})")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("ẽx [m]  (ẽy = 0, θ̃ = 0)")
    ax.set_title("value profile through the obstacle")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_value_slices.png"), dpi=150)
    plt.close(fig)


def fig_compute(timing, grid_meta):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    if timing:
        labs = [t["controller"] for t in timing]
        ax.barh(labs, [t["ms_mean"] for t in timing], height=0.5, color="#2a78d6")
        for i, t in enumerate(timing):
            ax.text(t["ms_mean"] * 1.1, i, f"{t['ms_mean']:.2f} ms (p95 {t['ms_p95']:.2f})", va="center",
                    fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
        ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
        ax.minorticks_off()
        ax.set_xlim(0.08, 40)
        ax.invert_yaxis()
        ax.set_xlabel("online time per control step [ms] (single process)")
        ax.set_title("Online computation")
    ax = axes[1]
    if grid_meta:  # clean single-job measurements from experiments.py offline_timing
        gm = [m for m in grid_meta if "RBF" not in m["grid"]]
        ns = [m["n_states"] for m in gm]
        ax.plot(ns, [m["sec_per_iter"] for m in gm], color="#2a78d6", marker="o", label="sec per GPI iteration")
        ax.plot(ns, [m["peak_gpu_MB"] / 1024 for m in gm], color="#eb6834", marker="s", label="peak GPU memory [GB]")
        ax.plot(ns, [m["dense_P_table_GB"] for m in gm], color=MUTED, marker="x", lw=1,
                label="dense P table (.., 8, 4) [GB]")
        for m in gm:
            ax.annotate(m["grid"], (m["n_states"], m["sec_per_iter"]), fontsize=7, color=INK2,
                        xytext=(4, 6), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("# discrete states (100 · nx · ny · nθ)")
        ax.set_title("Offline GPI cost vs grid size (σ×1, single job)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_compute.png"), dpi=150)
    plt.close(fig)


def fig_frontier(parts):
    """sigma x 1: collision steps per episode vs mean stage cost for every controller variant."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = [("CEC (margin m)", "#2a78d6", "o", lambda c: c.startswith("CEC"), "m"),
              ("GPI (penalty λ)", "#eb6834", "s", lambda c: c.startswith("GPI-lam"), "λ"),
              ("RBF averager", "#1baf7a", "^", lambda c: c in ("RBF", "RBF-smooth"), ""),
              ("GPI, deterministic model", "#eda100", "D", lambda c: c == "GPI-detmodel", "")]
    for lab, col, mk, sel, tag in groups:
        pts = [(r.stage_cost, r.coll_steps, r.controller) for _, r in parts.iterrows() if sel(r.controller)]
        if not pts:
            continue
        pts.sort()
        xs, ys, names = zip(*pts)
        ax.plot(xs, ys, color=col, marker=mk, ms=6, lw=1, label=lab, mec=SURF, mew=1)
        for x, y, n in pts:
            short = n.replace("CEC-m", "m=").replace("GPI-lam", "λ=").replace("RBF-smooth", "smooth")
            if short in ("RBF", "GPI-detmodel"):
                continue
            ax.annotate(short, (x, y), fontsize=7, color=INK2, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean stage cost ℓ per step (log)")
    ax.set_ylabel("collision steps per episode (mean of 200)")
    ax.set_title("σ×1 frontier: what each method pays for safety")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_frontier_k1.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
def cec_mechanism(rows, k):
    """For CEC collisions: did the plan predict x_{t+1} to be safe?"""
    onsets, planned_safe = 0, 0
    for r in rows:
        if r["controller"] != "CEC" or r["noise_scale"] != k:
            continue
        clr, pred = r["clearance"], r["pred_next_clear"]
        for t in range(len(pred)):
            if clr[t + 1] < 0 and clr[t] >= 0:
                onsets += 1
                planned_safe += int(pred[t] >= -1e-4)
    return onsets, planned_safe


def model_meta():
    out = {}
    d = os.path.join(RES, "models")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".npz"):
            m = json.loads(str(np.load(os.path.join(d, f))["meta"]))
            out[f[:-4]] = dict(name=f[:-4], n_states=m["n_states"], n_params=m["n_params"],
                               offline_sec=m["offline_sec"], peak_gpu_mb=m["peak_gpu_mb"],
                               iters=len(m["history"]), final_delta=m["history"][-1]["delta"],
                               final_policy_changed=m["history"][-1]["policy_changed"],
                               V_min_hist=min(h["V_mean"] for h in m["history"]))
    return out


def main():
    summary = {}
    meta = model_meta()
    summary["models"] = meta
    rows, df = load("main")
    if rows is not None:
        S = summarize(df)
        summary["main"] = S.to_dict(orient="records")
        tests = []
        for k in sorted(df.noise_scale.unique()):
            if k == 0:
                continue
            for ca, cb in (("CEC", "GPI"), ("GPI", "RBF"), ("GPI", "GPI-detmodel"), ("GPI", "GPI-lookup"),
                           ("CEC", "RBF"), ("GPI", "RBF-LS"), ("RBF", "RBF-LS")):
                row = dict(k=k, a=ca, b=cb)
                for col in ("mean_pos_err", "disc_cost", "mean_stage_cost", "min_clearance"):
                    A, B = paired(df, ca, cb, k, col)
                    if len(A) > 5 and np.any(A != B):
                        row[f"{col}_diff"] = float(np.mean(A - B))
                        row[f"{col}_p"] = float(stats.wilcoxon(A, B).pvalue)
                A, B = paired(df, ca, cb, k, "collided")
                p, n01, n10 = mcnemar(A, B)
                row.update(coll_p=p, coll_only_b=n01, coll_only_a=n10)
                tests.append(row)
        summary["tests"] = tests
        summary["cec_mechanism"] = {str(k): cec_mechanism(rows, k) for k in sorted(df.noise_scale.unique()) if k > 0}
        summary["n_passages"] = n_passages()
        fig_trajectories(rows)
        fig_corridor_zoom(rows)
        fig_noise_sweep(S)
        fig_tradeoff(S)
        fig_collision_timing(rows)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(S.drop(columns=[c for c in S.columns if c.endswith("_ci")]).round(4).to_string())
            print(pd.DataFrame(tests).round(4).to_string())
            print("CEC collision onsets (onsets, plan said safe):", summary["cec_mechanism"])
    parts = []
    for suite in ("pilot_margin", "pilot_lambda", "ablation2x2"):
        r2, d2 = load(suite)
        if r2 is not None:
            S2 = summarize(d2)
            summary[suite] = S2.to_dict(orient="records")
            parts.append(S2[S2.k == 1.0])
    if rows is not None:
        parts.append(S[(S.k == 1.0) & S.controller.isin(["RBF", "GPI-detmodel"])])
    r3, d3 = load("rbf")
    if r3 is not None:
        parts.append(summarize(d3))
    if parts:
        fig_frontier(pd.concat(parts, ignore_index=True))
    for suite in ("grid", "rbf"):
        r2, d2 = load(suite)
        if r2 is not None:
            S2 = summarize(d2)
            summary[suite] = S2.to_dict(orient="records")
            with pd.option_context("display.width", 250, "display.max_columns", 30):
                print(S2.drop(columns=[c for c in S2.columns if c.endswith("_ci")]).round(4).to_string())
    timing = None
    tp = os.path.join(RES, "timing.json")
    if os.path.exists(tp):
        timing = json.load(open(tp))
        summary["timing"] = timing
    op = os.path.join(RES, "offline_timing.json")
    grid_meta = json.load(open(op)) if os.path.exists(op) else None
    summary["offline_timing"] = grid_meta
    fig_compute(timing, grid_meta)
    fig_value_slices()
    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
