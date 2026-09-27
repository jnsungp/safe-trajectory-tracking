"""Supporting numbers quoted in the report that are not produced by analyze.py.
python extra_stats.py  ->  results/extra_stats.json"""
import json
import os

import numpy as np
from scipy import stats

import analyze as A
import common as C

RES = A.RES


def geometry():
    c1, c2, c3, c4 = C.OBSTACLES[:, :2]
    R = 0.5 + C.ROBOT_RADIUS
    r = C.REF[:, :2]
    d = np.linalg.norm(r[:, None] - C.OBSTACLES[None, :, :2], axis=2)
    inside = {f"C{i + 1}": np.where(d[:, i] < R)[0].tolist() for i in range(4)}
    # Monte Carlo: collision probability of one sample at the narrowest point of the C1-C3 corridor
    u = (c3 - c1) / np.linalg.norm(c3 - c1)
    gap = np.linalg.norm(c3 - c1) - 2 * R
    mid = c1 + u * (R + gap / 2)
    rng = np.random.default_rng(0)
    out = {}
    for off in (0.0, 0.1, 0.2):
        p = mid + np.array([-u[1], u[0]]) * off  # move along the corridor axis
        z = p + rng.standard_normal((1_000_000, 2)) * C.SIGMA[0]
        out[f"{off:.1f}"] = float((C.clearance(z) < 0).mean())
    return dict(corridor_gap_m=float(gap), ref_min_center_dist=d.min(0).round(4).tolist(),
                ref_inside_inflated=inside, c1_inflated_xmax=float(c1[0] + R),
                init_clearance=float(C.clearance(C.X_INIT)), init_error=float(np.linalg.norm(C.X_INIT[:2] - C.REF[0, :2])),
                p_coll_one_sample_by_offset=out)


def cec_horizon():
    from cec import CEC
    res = {}
    base = None
    for N in (5, 10, 15, 25):
        r = C.rollout(CEC(horizon=N), noise_scale=0.0, seed=0, keep_traj=True)
        if N == 10:
            base = r["traj"]
        res[N] = r
    return {N: dict(pos_err=float(r["mean_pos_err"]),
                    max_dev_vs_N10=float(np.linalg.norm(r["traj"][:, :2] - base[:, :2], axis=1).max()),
                    rms_dev_vs_N10=float(np.sqrt((np.linalg.norm(r["traj"][:, :2] - base[:, :2], axis=1) ** 2).mean())),
                    solver_fail=float(r["solver_fail_rate"]), slack=float(r["plan_slack_rate"]))
            for N, r in res.items()}


def ls_rbf_diagnostics():
    import torch
    from gpi import GPI
    out = {}
    for k in ("k0", "k0p25", "k0p5", "k1", "k2"):
        g = GPI.load(os.path.join(RES, "models", f"rbf_medium_{k}.npz"), device="cpu")
        V = g.V.grid_values()
        i0 = int(np.argmin(np.abs(g.grid.ex.numpy())))
        j0 = int(np.argmin(np.abs(g.grid.th.numpy())))
        out[k] = dict(frac_negative=float((V < 0).float().mean()), V_t0_e0=float(V[0, i0, i0, j0]))
    # ||Pi||_inf of the LS projection, estimated from 100 random rows (unit targets)
    g = GPI.load(os.path.join(RES, "models", "rbf_medium_k1.npz"), device="cpu")
    fv = g.V
    shape = fv.grid_values().shape
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(100):
        idx = tuple(int(rng.integers(n)) for n in shape)
        # row i of Pi = column i of Pi (symmetric): project a unit spike at node i
        y = torch.zeros(shape)
        y[idx] = 1.0
        fv.update_all(y)
        rows.append(float(fv.grid_values().abs().sum()))
    out["Pi_inf_norm_sampled_max"] = max(rows)
    out["Pi_inf_norm_sampled_median"] = float(np.median(rows))
    grid = GPI.load(os.path.join(RES, "models", "grid_medium_k1.npz"), device="cpu")
    i0 = int(np.argmin(np.abs(grid.grid.ex.numpy())))
    j0 = int(np.argmin(np.abs(grid.grid.th.numpy())))
    out["tabular_V_t0_e0"] = float(grid.V.grid_values()[0, i0, i0, j0])
    return out


def rollout_stats():
    rows, df = A.load("main")
    out = {}
    # free-space tracking error at noise x0: t >= 20 and reference clearance > 0.3 m
    rc = C.clearance(C.ref(np.arange(C.N_STEPS)))
    mask = (np.arange(C.N_STEPS) >= 20) & (rc > 0.3)
    fs = {}
    for r in rows:
        if r["noise_scale"] == 0.0:
            e = np.linalg.norm(r["traj"][:-1, :2] - C.ref(np.arange(C.N_STEPS))[:, :2], axis=1)
            fs[r["controller"]] = float(e[mask].mean())
    out["free_space_pos_err_k0"] = fs
    # sigma x 1, left lobe only (t mod 100 in [50, 100)): collision steps per episode
    left = {}
    for c in ("CEC", "GPI"):
        v = [np.sum((r["clearance"][:-1] < 0) & (np.arange(C.N_STEPS) % 100 >= 50))
             for r in rows if r["controller"] == c and r["noise_scale"] == 1.0]
        left[c] = float(np.mean(v))
    out["left_lobe_coll_steps_k1"] = left
    # break-even collision price, undiscounted and discounted
    be = {}
    disc = C.COST.gamma ** np.arange(C.N_STEPS + 1)
    for k in (0.25, 0.5, 1.0, 2.0):
        a = {r["seed"]: r for r in rows if r["controller"] == "CEC" and r["noise_scale"] == k}
        b = {r["seed"]: r for r in rows if r["controller"] == "GPI" and r["noise_scale"] == k}
        seeds = sorted(set(a) & set(b))
        dJ = np.mean([(b[s]["mean_stage_cost"] - a[s]["mean_stage_cost"]) * C.N_STEPS for s in seeds])
        dC = np.mean([a[s]["n_coll_steps"] - b[s]["n_coll_steps"] for s in seeds])
        dJd = np.mean([b[s]["disc_cost"] - a[s]["disc_cost"] for s in seeds])
        dCd = np.mean([(disc * (a[s]["clearance"] < 0)).sum() - (disc * (b[s]["clearance"] < 0)).sum() for s in seeds])
        be[str(k)] = dict(undiscounted=float(dJ / dC), discounted=float(dJd / dCd))
    out["break_even_price"] = be
    # onset predicted clearance (CEC)
    med = {}
    for k in (0.25, 0.5, 1.0, 2.0):
        preds = []
        for r in rows:
            if r["controller"] == "CEC" and r["noise_scale"] == k:
                clr, pr = r["clearance"], r["pred_next_clear"]
                for t in range(len(pr)):
                    if clr[t + 1] < 0 and clr[t] >= 0:
                        preds.append(pr[t])
        preds = np.array(preds)
        med[str(k)] = dict(median_pred=float(np.median(preds)), frac_below_1mm=float((np.abs(preds) < 1e-3).mean()))
    out["cec_onset_pred_clearance"] = med

    def mcn(ca, ka, cb, kb, rows_a=rows, rows_b=rows):
        A_ = {r["seed"]: r["collided"] for r in rows_a if r["controller"] == ca and r["noise_scale"] == ka}
        B_ = {r["seed"]: r["collided"] for r in rows_b if r["controller"] == cb and r["noise_scale"] == kb}
        s = sorted(set(A_) & set(B_))
        p, n01, n10 = A.mcnemar([A_[i] for i in s], [B_[i] for i in s])
        return dict(p=p, only_b=n01, only_a=n10)
    out["mcnemar_GPI_k1_vs_k2"] = mcn("GPI", 1.0, "GPI", 2.0)
    r3, _ = A.load("rbf")
    out["mcnemar_RBF_vs_RBFsmooth_k1"] = mcn("RBF", 1.0, "RBF-smooth", 1.0, rows, r3)
    return out


def more_rollout_stats():
    rows, _ = A.load("main")
    out = {}
    # CEC boundary riding: steps with |predicted next clearance| < 1 mm, and how often the next state collides
    br = {}
    for k in (0.25, 0.5, 1.0, 2.0):
        n_ride, n_coll, per_ep = 0, 0, []
        for r in rows:
            if r["controller"] == "CEC" and r["noise_scale"] == k:
                pr = np.asarray(r["pred_next_clear"])
                ride = np.abs(pr) < 1e-3
                per_ep.append(int(ride.sum()))
                n_ride += int(ride.sum())
                n_coll += int((ride & (r["clearance"][1:] < 0)).sum())
        br[str(k)] = dict(riding_steps_per_episode=float(np.mean(per_ep)), p_next_collision_given_riding=n_coll / max(n_ride, 1))
    out["cec_boundary_riding"] = br
    # CRN pairing strength: Spearman of episode min clearance, CEC vs GPI at sigma x 1
    a = {r["seed"]: r["min_clearance"] for r in rows if r["controller"] == "CEC" and r["noise_scale"] == 1.0}
    b = {r["seed"]: r["min_clearance"] for r in rows if r["controller"] == "GPI" and r["noise_scale"] == 1.0}
    s_ = sorted(set(a) & set(b))
    out["spearman_minclear_CEC_GPI_k1"] = float(stats.spearmanr([a[i] for i in s_], [b[i] for i in s_])[0])
    # GPI at sigma x 2: nearest obstacle of collision states
    near = {}
    for r in rows:
        if r["controller"] == "GPI" and r["noise_scale"] == 2.0:
            for t in np.where(r["clearance"] < 0)[0]:
                d = np.linalg.norm(r["traj"][t, :2] - C.OBSTACLES[:, :2], axis=1)
                key = f"C{int(np.argmin(d)) + 1}"
                near[key] = near.get(key, 0) + 1
    out["gpi_k2_collision_nearest_obstacle"] = near
    # per-passage collision steps by lobe for GPI lambda=100 (corridor on both sides)
    rl, _ = A.load("pilot_lambda")
    lobe = {"right": [], "left": []}
    for r in rl:
        if r["controller"] == "GPI-lam100":
            coll = r["clearance"][:-1] < 0
            tm = np.arange(C.N_STEPS) % 100
            lobe["right"].append(coll[(tm >= 5) & (tm < 45)].sum() / 3)  # 3 right passages per episode
            lobe["left"].append(coll[(tm >= 55) & (tm < 95)].sum() / 2)  # 2 left passages per episode
    out["lam100_coll_steps_per_passage"] = {k: float(np.mean(v)) for k, v in lobe.items()}
    # CEC margin 10 cm: mean forward speed
    rm, _ = A.load("pilot_margin")
    out["cec_m0.1_mean_v"] = {str(k): float(np.mean([r["controls"][:, 0].mean() for r in rm
                                                     if r["controller"] == "CEC-m0.1" and r["noise_scale"] == k]))
                              for k in (0.25, 1.0)}
    return out


if __name__ == "__main__":
    out = dict(geometry=geometry(), cec_horizon=cec_horizon(), ls_rbf=ls_rbf_diagnostics(), rollouts=rollout_stats(),
               rollouts_more=more_rollout_stats())
    with open(os.path.join(RES, "extra_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
