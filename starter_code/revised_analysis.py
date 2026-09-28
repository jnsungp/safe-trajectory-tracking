"""Analyze the corrected-risk experiment without overwriting the original report data.

Run after `experiments.py rollout --suite revised` and `--suite revised_ablation`.
"""
import json
import math
import os
import pickle

import numpy as np
from scipy.stats import ncx2, norm

import analyze as A
import common as C


def survival_metrics(rows, controller, k):
    selected = [r for r in rows if r["controller"] == controller and r["noise_scale"] == k]
    n = len(selected)
    hit = np.asarray([r["first_coll_t"] for r in selected])
    safe_steps = np.where(hit < 0, C.N_STEPS, hit)
    by_100 = int(((hit >= 0) & (hit <= 100)).sum())
    by_200 = int(((hit >= 0) & (hit <= 200)).sum())
    return dict(safe_steps_mean=float(safe_steps.mean()),
                safe_steps_mean_ci=A.boot_ci(safe_steps),
                safe_steps_median=float(np.median(safe_steps)),
                first_hit_median=float(np.median(hit[hit >= 0])) if (hit >= 0).any() else None,
                p_hit_100=by_100 / n, p_hit_100_ci=A.wilson(by_100, n),
                p_hit_200=by_200 / n, p_hit_200_ci=A.wilson(by_200, n),
                pre_hit_stage_cost=float(np.nanmean([r["pre_hit_stage_cost"] for r in selected])))


def risk_at_corridor():
    a, b = C.OBSTACLES[[0, 2], :2]
    midpoint = (a + b) / 2
    distance = np.linalg.norm(C.OBSTACLES[:, :2] - midpoint, axis=1)
    radius = C.OBSTACLES[:, 2] + C.ROBOT_RADIUS
    out = []
    for k in (0.25, 0.5, 1.0, 2.0):
        sigma = C.SIGMA[0] * k
        per_disk_exact = ncx2.cdf((radius / sigma) ** 2, 2, (distance / sigma) ** 2)
        per_disk_halfplane = norm.cdf((radius - distance) / sigma)
        out.append(dict(k=k, exact_disk_union=float(per_disk_exact.sum()),
                        disjoint_halfplane_sum=float(per_disk_halfplane.sum()),
                        old_independent_halfplane=float(1 - np.prod(1 - per_disk_halfplane))))
    return out


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    rows, df = A.load("revised")
    if rows is None:
        raise FileNotFoundError("Run experiments.py rollout --suite revised first")
    summary = A.summarize(df)
    for idx, rec in summary.iterrows():
        summary.loc[idx, "safe_steps_mean"] = survival_metrics(rows, rec.controller, rec.k)["safe_steps_mean"]
    main_rows = summary.to_dict(orient="records")
    for rec in main_rows:
        rec.update(survival_metrics(rows, rec["controller"], rec["k"]))

    result = dict(protocol=dict(n_seeds=200, n_steps=C.N_STEPS, risk_mode="exact_disks",
                                rbf_online_value="interpolated", risk_lookup_step_m=0.0005),
                  corridor_risk=risk_at_corridor(), main=main_rows,
                  models={k: v for k, v in A.model_meta().items() if k.startswith("revised_")})
    reference_clearance = C.clearance(C.ref(np.arange(C.N_STEPS)))
    free_mask = (np.arange(C.N_STEPS) >= 20) & (reference_clearance > 0.3)
    result["free_space_pos_err_k0"] = {
        r["controller"]: float(np.linalg.norm(
            r["traj"][:-1, :2] - C.ref(np.arange(C.N_STEPS))[:, :2], axis=1)[free_mask].mean())
        for r in rows if r["noise_scale"] == 0.0
    }
    old_rows, _ = A.load("main")
    if old_rows is not None:
        comparison = []
        for k in (0.25, 0.5, 1.0, 2.0):
            for controller in ("GPI", "RBF"):
                old = {r["seed"]: r for r in old_rows if r["controller"] == controller and r["noise_scale"] == k}
                new = {r["seed"]: r for r in rows if r["controller"] == controller and r["noise_scale"] == k}
                seeds = sorted(set(old) & set(new))
                if not seeds:
                    continue
                a = [old[s]["collided"] for s in seeds]
                b = [new[s]["collided"] for s in seeds]
                p, new_only, old_only = A.mcnemar(a, b)
                comparison.append(dict(k=k, controller=controller, n=len(seeds),
                                       old_collisions=int(sum(a)), revised_collisions=int(sum(b)),
                                       old_only=old_only, revised_only=new_only, mcnemar_p=p,
                                       old_pos_err=float(np.mean([old[s]["mean_pos_err"] for s in seeds])),
                                       revised_pos_err=float(np.mean([new[s]["mean_pos_err"] for s in seeds]))))
        result["paired_old_new"] = comparison
    ab_rows, ab_df = A.load("revised_ablation")
    if ab_rows is not None:
        result["ablation"] = A.summarize(ab_df).to_dict(orient="records")
        ab_tests = []
        for a_name, b_name in (("T-noise+risk4cm", "T-det+risk4cm"),
                               ("T-noise+risk1cm", "T-noise+risk4cm"),
                               ("T-det+risk1cm", "T-det+risk4cm")):
            a = {r["seed"]: r["collided"] for r in ab_rows if r["controller"] == a_name}
            b = {r["seed"]: r["collided"] for r in ab_rows if r["controller"] == b_name}
            seeds = sorted(set(a) & set(b))
            p, b_only, a_only = A.mcnemar([a[s] for s in seeds], [b[s] for s in seeds])
            ab_tests.append(dict(a=a_name, b=b_name, n=len(seeds), p=p,
                                 a_only=a_only, b_only=b_only))
        result["ablation_tests"] = ab_tests
    lam_rows, lam_df = A.load("revised_lambda")
    if lam_rows is not None:
        result["lambda_sweep"] = A.summarize(lam_df).to_dict(orient="records")
    eval_rows, eval_df = A.load("rbf_evaluator_ablation")
    if eval_rows is not None:
        result["rbf_evaluator_ablation"] = A.summarize(eval_df).to_dict(orient="records")
        paired_eval = []
        for k in (0.5, 1.0):
            a = {r["seed"]: r for r in eval_rows if r["noise_scale"] == k and r["controller"] == "RBF-legacy-kernel"}
            b = {r["seed"]: r for r in eval_rows if r["noise_scale"] == k and r["controller"] == "RBF-legacy-interpolated"}
            seeds = sorted(set(a) & set(b))
            p, interp_only, kernel_only = A.mcnemar([a[s]["collided"] for s in seeds],
                                                     [b[s]["collided"] for s in seeds])
            paired_eval.append(dict(k=k, n=len(seeds), kernel_only=kernel_only,
                                    interpolated_only=interp_only, mcnemar_p=p))
        result["rbf_evaluator_paired"] = paired_eval
    for name in ("revised_timing", "revised_offline_timing"):
        path = os.path.join(A.RES, name + ".json")
        if os.path.exists(path):
            with open(path) as fh:
                result[name] = json.load(fh)
    phase_path = os.path.join(A.RES, "revised_phase.json")
    if os.path.exists(phase_path):
        with open(phase_path) as fh:
            result["phase_robustness"] = json.load(fh)
        raw_phase_path = os.path.join(A.RES, "rollouts_revised_phase.pkl")
        if os.path.exists(raw_phase_path):
            with open(raw_phase_path, "rb") as fh:
                phase_rows = pickle.load(fh)
            phase_tests = []
            for controller in ("GPI", "RBF"):
                hits = {(r["phase"], r["seed"]): r["collided"] for r in phase_rows
                        if r["controller"] == controller}
                for a in (0, 25, 50):
                    for b in (25, 50, 75):
                        if b <= a:
                            continue
                        x = [hits[a, seed] for seed in range(200)]
                        y = [hits[b, seed] for seed in range(200)]
                        p, b_only, a_only = A.mcnemar(x, y)
                        phase_tests.append(dict(controller=controller, a=a, b=b, p=p,
                                                a_only=a_only, b_only=b_only))
            result["phase_tests"] = phase_tests
    output = os.path.join(A.RES, "revised_summary.json")
    with open(output, "w") as fh:
        json.dump(json_safe(result), fh, indent=2, allow_nan=False)
    print(output)
    for rec in main_rows:
        print(rec["controller"], rec["k"], "P(hit)", rec["p_coll"],
              "safe steps", round(rec["safe_steps_mean"], 1),
              "position error", round(rec["pos_err"], 3))


if __name__ == "__main__":
    main()
