"""Aggregate statistics for the follow-up experiment -> results/followup_summary.json.

Run after:
    python experiments.py rollout --suite riskcec
    python experiments.py rollout --suite path
    python experiments.py rollout --suite path_frontier
    python experiments.py rollout --suite heldout
    python experiments.py timing_followup

Two collision counts are reported for every controller. "Sampled" counts a collision when a sampled
position x_t is inside an inflated obstacle (the definition used in the rest of the study). "Path" also
checks the motion between samples: the noise-free arc of each step with the step's noise blended in
linearly, at 21 points per step (analyze.arc_clearance).
"""
import json
import os
import pickle

import numpy as np
from scipy.stats import ncx2, norm

import analyze as A
import common as C
from revised_analysis import json_safe

GAP_PAIRS = A.CORRIDORS  # (C1, C3) and (C2, C4)


def arc_points(traj, controls, n_sub=21, blend=True):
    """Positions along each step: the noise-free unicycle arc from x_t with the step's noise blended in
    linearly (same construction as analyze.arc_clearance). blend=False gives the commanded arc alone,
    which ends at f(x_t, u_t) instead of x_{t+1}. Returns px, py of shape (steps, n_sub)."""
    X, U = traj[:-1], controls
    Wn = traj[1:] - C.f(X, U)
    Wn[:, 2] = C.wrap(Wn[:, 2])
    if not blend:
        Wn = np.zeros_like(Wn)
    s = np.linspace(0, C.DT, n_sub)[None, :]
    v, om, th = U[:, :1], U[:, 1:2], X[:, 2:3]
    half = om * s / 2
    step = s * np.sinc(half / np.pi) * v
    px = X[:, :1] + step * np.cos(th + half) + (s / C.DT) * Wn[:, :1]
    py = X[:, 1:2] + step * np.sin(th + half) + (s / C.DT) * Wn[:, 1:2]
    return px, py


def path_clearance_per_step(traj, controls, n_sub=21):
    """Smallest clearance along each step's path."""
    px, py = arc_points(traj, controls, n_sub)
    return C.clearance(np.stack([px, py], -1)).min(axis=1)


def gap_crossings(traj, controls):
    """Steps whose chord crosses a gap's center segment, with the commanded speed of that step."""
    out = []
    p, q = traj[:-1, :2], traj[1:, :2]
    for i, j in GAP_PAIRS:
        hit = A._seg_cross(p, q, C.OBSTACLES[i, :2], C.OBSTACLES[j, :2])
        for t in np.flatnonzero(hit):
            approach = controls[max(t - 3, 0):t, 0]  # the three steps before the crossing step
            out.append(dict(t=int(t), gap="C1-C3" if i == 0 else "C2-C4", v=float(controls[t, 0]),
                            v_approach=float(approach.mean()) if len(approach) else None))
    return out


def dash_example():
    """Risk of one full-speed step through the C1-C3 gap center (from 26 cm before to 24 cm after it),
    scored at the next sample only and along the arc (4 points), as GPI's two risk terms score it."""
    import torch
    from gpi import GPI, GpiConfig
    mid = (C.OBSTACLES[0, :2] + C.OBSTACLES[2, :2]) / 2
    d = C.OBSTACLES[2, :2] - C.OBSTACLES[0, :2]
    along = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    x = np.r_[mid - 0.26 * along, np.arctan2(along[1], along[0])]
    u = np.array([1.0, 0.0])
    out = []
    for k in (0.5, 1.0):
        risks = {}
        for m in (1, 4):
            g = GPI(GpiConfig(ex_space=np.array([-1., 0., 1.]), ey_space=np.array([-1., 0., 1.]),
                              eth_space=np.linspace(-np.pi, np.pi, 4, endpoint=False), v_space=np.array([0.1, 1.0]),
                              w_space=np.array([-1., 1.]), risk_mode="exact_disks", noise_scale=k,
                              risk_substeps=m, device="cpu"))
            e = torch.tensor(C.error_state(0, x), dtype=torch.float32).view(1, 1, 3)
            U = torch.tensor(u, dtype=torch.float32).view(1, 1, 2)
            tt = torch.tensor([0])
            mx, my, _ = g._next_mean(tt, e, U)
            px, py = mx + g.ref[1, 0], my + g.ref[1, 1]
            risks[m] = float(g.path_risk(tt, e, U, px, py))
        out.append(dict(k=k, sample_risk=risks[1], path_risk=risks[4], ratio=risks[4] / risks[1],
                        sample_clearances=[float(C.clearance(x)), float(C.clearance(C.f(x, u)))]))
    return out


def risk_calibration(rows, controller, k, model, max_episodes=200):
    """Path risk that a GPI model assigns to the steps it executed, against the realized path collisions,
    at gap-crossing steps and in bins of the predicted risk."""
    import torch
    from gpi import GPI
    g = GPI.load(os.path.join(A.RES, "models", f"{model}.npz"), device="cpu")
    pred, real, at_gap = [], [], []
    for r in [r for r in rows if r["controller"] == controller and r["noise_scale"] == k][:max_episodes]:
        X, U = r["traj"], r["controls"]
        t = np.arange(len(U))
        E = torch.tensor(np.stack([C.error_state(i, X[i]) for i in t]), dtype=torch.float32)[:, None]
        tt = torch.tensor(t % C.T_PERIOD)
        Ut = torch.tensor(U, dtype=torch.float32)[:, None]
        with torch.no_grad():
            mx, my, _ = g._next_mean(tt, E, Ut)
            tn = (tt + 1) % C.T_PERIOD
            p = g.path_risk(tt, E, Ut, mx + g.ref[tn, 0][:, None], my + g.ref[tn, 1][:, None]).numpy().ravel()
        cross = {c["t"] for c in r["gap_crossings"]}
        pred += list(p)
        real += list(path_clearance_per_step(X, U) < 0)
        at_gap += [i in cross for i in t]
    pred, real, at_gap = np.array(pred), np.array(real), np.array(at_gap)
    bins = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    out = dict(controller=controller, k=k, model=model, n_steps=int(len(pred)),
               gap_steps=int(at_gap.sum()), gap_pred=float(pred[at_gap].mean()) if at_gap.any() else None,
               gap_real=float(real[at_gap].mean()) if at_gap.any() else None, bins=[])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.any():
            out["bins"].append(dict(lo=lo, hi=hi, n=int(m.sum()), pred=float(pred[m].mean()), real=float(real[m].mean())))
    return out


def load_rows(suite):
    path = os.path.join(A.RES, f"rollouts_{suite}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        rows = pickle.load(fh)
    for r in rows:
        clr = path_clearance_per_step(r["traj"], r["controls"])
        hit = np.flatnonzero(clr < 0)
        r["path_first_hit"] = int(hit[0]) if len(hit) else -1
        r["path_collided"] = bool(len(hit))
        r["path_min_clearance"] = float(clr.min())
        r["gap_crossings"] = gap_crossings(r["traj"], r["controls"])
        px, py = arc_points(r["traj"], r["controls"], blend=False)
        r["commanded_arc_collided"] = bool((C.clearance(np.stack([px, py], -1)) < 0).any())
        cross_t = {c["t"] for c in r["gap_crossings"]}
        r["path_first_hit_at_gap"] = bool(len(hit)) and any(abs(hit[0] - t) <= 1 for t in cross_t)
        err = np.linalg.norm(r["traj"][:-1, :2] - C.ref(r.get("start_t", 0) + np.arange(len(r["controls"])))[:, :2], axis=1)
        r["max_pos_err"] = float(err.max())
    return rows


def summarize(rows, controller, k):
    g = [r for r in rows if r["controller"] == controller and r["noise_scale"] == k]
    n = len(g)
    if n == 0:
        return None
    coll = np.array([r["collided"] for r in g])
    pcoll = np.array([r["path_collided"] for r in g])
    first = np.array([r["first_coll_t"] for r in g])
    pfirst = np.array([r["path_first_hit"] for r in g])
    err = np.array([r["mean_pos_err"] for r in g])
    crossings = [c for r in g for c in r["gap_crossings"]]
    out = dict(controller=controller, k=k, n=n,
               p_coll=float(coll.mean()), p_coll_ci=A.wilson(int(coll.sum()), n),
               p_path=float(pcoll.mean()), p_path_ci=A.wilson(int(pcoll.sum()), n),
               p_commanded_arc=float(np.mean([r["commanded_arc_collided"] for r in g])),
               pos_err=float(err.mean()), pos_err_ci=A.boot_ci(err),
               stage_cost=float(np.mean([r["mean_stage_cost"] for r in g])),
               safe_steps_mean=float(np.where(first < 0, C.N_STEPS, first).mean()),
               path_safe_steps_mean=float(np.where(pfirst < 0, C.N_STEPS, pfirst).mean()),
               min_clear_med=float(np.median([r["min_clearance"] for r in g])),
               path_min_clear_med=float(np.median([r["path_min_clearance"] for r in g])),
               corridor_frac=float(np.mean([len(r["gap_crossings"]) for r in g]) / A.n_passages()),
               gap_speed_mean=float(np.mean([c["v"] for c in crossings])) if crossings else None,
               gap_speed_full_frac=float(np.mean([c["v"] > 0.95 for c in crossings])) if crossings else None,
               gap_approach_speed=float(np.mean([c["v_approach"] for c in crossings if c["v_approach"] is not None]))
               if crossings else None,
               n_gap_crossings=len(crossings),
               max_err_over_1m=float(np.mean([r["max_pos_err"] > 1.0 for r in g])),
               n_path_only=int(np.sum(pcoll & ~coll)),
               n_path_only_first_at_gap=int(np.sum(pcoll & ~coll & np.array([r["path_first_hit_at_gap"] for r in g]))),
               n_first_path_hit_at_gap=int(np.sum([r["path_first_hit_at_gap"] for r in g])),
               ctrl_ms=float(np.mean([r["ctrl_ms_mean"] for r in g])))
    for extra in ("solver_fail_rate", "plan_slack_rate", "policy_seed_rate", "solver_iters_mean"):
        vals = [r[extra] for r in g if extra in r]
        if vals:
            out[extra] = float(np.mean(vals))
    return out


def summarize_all(rows):
    keys = sorted({(r["controller"], r["noise_scale"]) for r in rows}, key=lambda x: (x[0], x[1]))
    return [summarize(rows, c, k) for c, k in keys]


def paired_test(rows_a, name_a, rows_b, name_b, k, key="path_collided"):
    a = {r["seed"]: r[key] for r in rows_a if r["controller"] == name_a and r["noise_scale"] == k}
    b = {r["seed"]: r[key] for r in rows_b if r["controller"] == name_b and r["noise_scale"] == k}
    seeds = sorted(set(a) & set(b))
    p, b_only, a_only = A.mcnemar([a[s] for s in seeds], [b[s] for s in seeds])
    return dict(a=name_a, b=name_b, k=k, key=key, n=len(seeds), p=p, a_only=a_only, b_only=b_only)


def time_series(rows, controller, k):
    """Mean and 90th percentile of the position error over time (for the stall figure)."""
    g = [r for r in rows if r["controller"] == controller and r["noise_scale"] == k]
    E = np.array([np.linalg.norm(r["traj"][:-1, :2] - C.ref(np.arange(len(r["controls"])))[:, :2], axis=1)
                  for r in g])
    return dict(mean=E.mean(0).tolist(), p90=np.percentile(E, 90, axis=0).tolist())


def main():
    out = dict(protocol=dict(n_seeds=200, heldout_seeds=[1000, 1199], n_steps=C.N_STEPS, path_points_per_step=21,
                             risk_substeps=4, horizon=10))
    # a per-step chance constraint P(hit) <= delta with a half-plane bound tightens each obstacle by
    # sigma * Phi^-1(1 - delta): the CEC margins of the earlier sweep, read as risk bounds at noise x1
    out["margin_as_chance_bound"] = [dict(margin_m=m, per_step_risk=float(norm.sf(m / C.SIGMA[0])))
                                     for m in (0.025, 0.05, 0.1)]
    a, b = C.OBSTACLES[[0, 2], :2]
    dist = np.linalg.norm(C.OBSTACLES[:, :2] - (a + b) / 2, axis=1)
    radius = C.OBSTACLES[:, 2] + C.ROBOT_RADIUS
    out["gap_center_risk_k1"] = float(ncx2.cdf((radius / C.SIGMA[0]) ** 2, 2, (dist / C.SIGMA[0]) ** 2).sum())
    out["dash_example"] = dash_example()
    rev_path = os.path.join(A.RES, "revised_summary.json")
    if os.path.exists(rev_path):  # results of the main experiment that the follow-up text refers to
        with open(rev_path) as fh:
            rev = json.load(fh)
        out["referenced_main_results"] = dict(
            ablation_collision_rate={r["controller"]: r["p_coll"] for r in rev["ablation"]
                                     if r["controller"] in ("T-det+risk4cm", "T-noise+risk4cm")})
    # existing controllers (next-sample risk), re-scored along the path
    base = []
    for kn in ("k0", "k0p25", "k0p5", "k1", "k2"):
        rows = load_rows(f"revised_part_{kn}")
        if rows is not None:
            base += rows
    out["baseline"] = summarize_all(base)
    lam_rows = load_rows("revised_lambda")
    if lam_rows is not None:
        out["baseline_lambda"] = summarize_all(lam_rows)
    margin_rows = load_rows("pilot_margin")
    if margin_rows is not None:
        out["baseline_margin"] = [s for s in summarize_all(margin_rows) if s["k"] == 1.0]
    risk_rows = load_rows("riskcec")
    if risk_rows is not None:
        out["riskcec"] = summarize_all(risk_rows)
        out["riskcec_error_k1"] = {c: time_series(risk_rows, c, 1.0) for c in ("RiskCEC-soft", "RiskCEC-hard",
                                                                                 "RiskCEC-V")}
        out["riskcec_error_k1"]["GPI"] = time_series(base, "GPI", 1.0)
        out["riskcec_error_k0p5"] = {"RiskCEC-V": time_series(risk_rows, "RiskCEC-V", 0.5),
                                     "GPI": time_series(base, "GPI", 0.5)}
        out["riskcec_error_peaks"] = {
            key: {f"k{k:g}": float(np.max(time_series(src, c, k)["mean"])) for k in (0.5, 1.0)}
            for key, c, src in (("RiskCEC-V", "RiskCEC-V", risk_rows), ("RiskCEC-hard", "RiskCEC-hard", risk_rows),
                                ("GPI", "GPI", base))}
        out["riskcec_tests"] = [paired_test(risk_rows, "RiskCEC-V", base, "GPI", k, "collided")
                                for k in (0.25, 0.5, 1.0, 2.0)]
    path_rows = load_rows("path")
    if path_rows is not None:
        out["path"] = summarize_all(path_rows)
        out["path_tests"] = [paired_test(path_rows, "Hybrid", path_rows, "GPI-path", k) for k in (0.25, 0.5, 1.0, 2.0)]
        out["path_tests"] += [paired_test(path_rows, "RiskCEC-path-V", path_rows, "Hybrid", 1.0)]
        out["path_risk_calibration"] = [risk_calibration(path_rows, "GPI-path", k, f"path_grid_medium_{kn}")
                                        for k, kn in ((0.25, "k0p25"), (0.5, "k0p5"))]
        # tracking difference between the hybrid and path-aware GPI on the same seeds
        diffs = []
        for k in (0.25, 0.5, 1.0, 2.0):
            a = {r["seed"]: r["mean_pos_err"] for r in path_rows if r["controller"] == "Hybrid" and r["noise_scale"] == k}
            b = {r["seed"]: r["mean_pos_err"] for r in path_rows if r["controller"] == "GPI-path" and r["noise_scale"] == k}
            d = np.array([a[s] - b[s] for s in sorted(set(a) & set(b))])
            diffs.append(dict(k=k, n=len(d), mean=float(d.mean()), ci=A.boot_ci(d), frac_better=float((d < 0).mean())))
        out["path_err_diff"] = diffs
    fr_rows = load_rows("path_frontier")
    if fr_rows is not None:
        out["path_frontier"] = summarize_all(fr_rows)
        # same objective (lambda), same seeds: does the hybrid match GPI's path safety, and how does it track?
        pairs = [("Hybrid", "GPI-path", path_rows)] if path_rows is not None else []
        pairs += [(f"Hybrid-lam{lam}", f"GPI-path-lam{lam}", fr_rows) for lam in (10, 100, 10000)]
        matched = []
        for a_name, b_name, rows in pairs:
            test = paired_test(rows, a_name, rows, b_name, 1.0)
            a = {r["seed"]: r["mean_pos_err"] for r in rows if r["controller"] == a_name and r["noise_scale"] == 1.0}
            b = {r["seed"]: r["mean_pos_err"] for r in rows if r["controller"] == b_name and r["noise_scale"] == 1.0}
            d = np.array([a[s] - b[s] for s in sorted(set(a) & set(b))])
            test.update(err_diff_mean=float(d.mean()), err_diff_ci=A.boot_ci(d), frac_better=float((d < 0).mean()))
            matched.append(test)
        out["matched_lambda"] = matched
    det_rows = load_rows("path_det")
    if det_rows is not None:
        out["path_det"] = summarize_all(det_rows)
        if path_rows is not None:
            out["path_det_tests"] = [paired_test(det_rows, "GPI-path-det", path_rows, "GPI-path", k) for k in (1.0, 2.0)]
            out["path_det_tests"].append(paired_test(det_rows, "Hybrid-planonly", path_rows, "Hybrid", 2.0))
    ho_rows = load_rows("heldout")
    if ho_rows is not None:
        out["heldout"] = summarize_all(ho_rows)
        out["heldout_tests"] = [paired_test(ho_rows, "Hybrid", ho_rows, "GPI-path", 1.0),
                                paired_test(ho_rows, "GPI-path", ho_rows, "GPI", 1.0)]
    tpath = os.path.join(A.RES, "followup_timing.json")
    if os.path.exists(tpath):
        with open(tpath) as fh:
            out["timing"] = json.load(fh)
    meta = A.model_meta()
    out["models"] = {k: v for k, v in meta.items() if k.startswith("path_")}
    dest = os.path.join(A.RES, "followup_summary.json")
    with open(dest, "w") as fh:
        json.dump(json_safe(out), fh, indent=2, allow_nan=False)
    print(dest)
    for sec in ("baseline", "riskcec", "path", "path_frontier", "heldout"):
        for s in out.get(sec, []):
            print(f"{sec:14s} {s['controller']:18s} k={s['k']:<5g} sampled {100 * s['p_coll']:5.1f}%  "
                  f"path {100 * s['p_path']:5.1f}%  err {100 * s['pos_err']:5.1f} cm  gaps {s['corridor_frac']:.2f}  "
                  f"gap v {s['gap_speed_mean'] if s['gap_speed_mean'] is None else round(s['gap_speed_mean'], 2)}  "
                  f"ms {s['ctrl_ms']:.1f}")


if __name__ == "__main__":
    main()
