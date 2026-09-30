"""Experiments for CEC vs tabular GPI vs RBF GPI.

    python experiments.py train  --models grid_medium_k1 rbf_medium_k1 ...   (GPU, offline GPI)
    python experiments.py rollout --suite main|grid|rbf|pilot_margin|pilot_lambda (CPU, multiprocess)
    python experiments.py timing                                              (single-process timing)

All rollouts use common random numbers: episode `seed` uses the same standard-normal draws for
every controller and every noise scale (common.noise_sequence).
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import get_context
from time import perf_counter

import numpy as np

import common as C

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR = os.path.join(RESULTS, "models")
NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0]
N_SEEDS = 200


def _kname(k):
    return f"k{k:g}".replace(".", "p")


# ------------------------------------------------------------------------------------------
# Model zoo (offline GPI)
# ------------------------------------------------------------------------------------------
def model_specs():
    specs = {}
    for k in NOISE_LEVELS:
        specs[f"grid_medium_{_kname(k)}"] = dict(value_type="grid", grid="medium", noise_scale=k)
        specs[f"rbf_medium_{_kname(k)}"] = dict(value_type="rbf", grid="medium", noise_scale=k,
                                                  rbf_stride_t=1, rbf_stride_e=2, rbf_ls_t=0.8, rbf_ls_e=1.6)
        # normalized-RBF averager (stable projected GPI); the "rbf_*" models above are plain LS-RBF
        specs[f"rbfavg_medium_{_kname(k)}"] = dict(value_type="rbf", grid="medium", noise_scale=k, rbf_fit="avg",
                                                     rbf_stride_t=1, rbf_stride_e=2, rbf_ls_t=0.8, rbf_ls_e=1.6)
    specs["grid_coarse_k1"] = dict(value_type="grid", grid="coarse", noise_scale=1.0)
    specs["grid_fine_k1"] = dict(value_type="grid", grid="fine", noise_scale=1.0)
    # smoother / more compressed RBF (dose-response for H5)
    specs["rbf_smooth_k1"] = dict(value_type="rbf", grid="medium", noise_scale=1.0,
                                  rbf_stride_t=2, rbf_stride_e=3, rbf_ls_t=1.6, rbf_ls_e=3.0)
    # pilot B: collision-penalty sweep (lambda) for the tracking-vs-safety frontier
    for lam in (10, 100, 10000):
        specs[f"grid_medium_k1_lam{lam}"] = dict(value_type="grid", grid="medium", noise_scale=1.0,
                                                 collision_penalty=float(lam))
    # 2x2 ablation at sigma x 1: transition noise (on/off) x risk-term sigma (true 4 cm / 1 cm floor)
    specs["grid_medium_k1_risk1cm"] = dict(value_type="grid", grid="medium", noise_scale=1.0, risk_sigma=0.01)
    specs["grid_medium_k0_risk4cm"] = dict(value_type="grid", grid="medium", noise_scale=0.0, risk_sigma=0.04)
    specs["rbfavg_smooth_k1"] = dict(value_type="rbf", grid="medium", noise_scale=1.0, rbf_fit="avg",
                                     rbf_stride_t=2, rbf_stride_e=3, rbf_ls_t=1.6, rbf_ls_e=3.0)
    # Revised experiment: exact 2-D disk probabilities and a consistent RBF evaluator.
    for k in NOISE_LEVELS:
        kn = _kname(k)
        specs[f"revised_grid_medium_{kn}"] = dict(value_type="grid", grid="medium", noise_scale=k,
                                                  risk_mode="exact_disks")
        specs[f"revised_rbfavg_medium_{kn}"] = dict(value_type="rbf", grid="medium", noise_scale=k,
                                                    risk_mode="exact_disks", rbf_fit="avg",
                                                    rbf_online_value="interpolated", rbf_stride_t=1,
                                                    rbf_stride_e=2, rbf_ls_t=0.8, rbf_ls_e=1.6)
    specs["revised_grid_medium_k1_risk1cm"] = dict(value_type="grid", grid="medium", noise_scale=1.0,
                                                     risk_sigma=0.01, risk_mode="exact_disks")
    specs["revised_grid_medium_k0_risk4cm"] = dict(value_type="grid", grid="medium", noise_scale=0.0,
                                                     risk_sigma=0.04, risk_mode="exact_disks")
    for lam in (10, 100, 10000):
        specs[f"revised_grid_medium_k1_lam{lam}"] = dict(value_type="grid", grid="medium",
                                                          noise_scale=1.0, risk_mode="exact_disks",
                                                          collision_penalty=float(lam))
    # plain least-squares RBF with the same risk and evaluator (value-function comparison)
    specs["revised_rbf_medium_k1"] = dict(value_type="rbf", grid="medium", noise_scale=1.0, risk_mode="exact_disks",
                                          rbf_fit="ls", rbf_online_value="interpolated", rbf_stride_t=1,
                                          rbf_stride_e=2, rbf_ls_t=0.8, rbf_ls_e=1.6)
    # Path-aware risk: the largest collision probability along each step's arc (4 points per step)
    for k in NOISE_LEVELS:
        specs[f"path_grid_medium_{_kname(k)}"] = dict(value_type="grid", grid="medium", noise_scale=k,
                                                     risk_mode="exact_disks", risk_substeps=4)
    for lam in (10, 100, 10000):
        specs[f"path_grid_medium_k1_lam{lam}"] = dict(value_type="grid", grid="medium", noise_scale=1.0,
                                                      risk_mode="exact_disks", risk_substeps=4,
                                                      collision_penalty=float(lam))
    # same path risk, noise-free transitions (the DP counterpart of CEC's noise-free predictions)
    for k in (1.0, 2.0):
        specs[f"path_grid_medium_{_kname(k)}_det"] = dict(value_type="grid", grid="medium", noise_scale=0.0,
                                                          risk_sigma=0.04 * k, risk_mode="exact_disks",
                                                          risk_substeps=4)
    for name, spec in specs.items():
        if not name.startswith(("revised_", "path_")):
            spec.setdefault("risk_mode", "independent_halfplane")
            spec.setdefault("rbf_online_value", "kernel")
    return specs


def train(names, num_iters=20, tol=0.05, max_gpu_gb=4.0):
    import torch
    from gpi import GPI, GpiConfig, GRID_PRESETS, theta_grid
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_gb / total))
    specs = model_specs()
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in names:
        out = os.path.join(MODELS_DIR, f"{name}.npz")
        if os.path.exists(out):
            print(f"[skip] {name} exists", flush=True)
            continue
        s = dict(specs[name])
        pgrid, nth = GRID_PRESETS[s.pop("grid")]
        cfg = GpiConfig(ex_space=pgrid, ey_space=pgrid, eth_space=theta_grid(nth), device="cuda", **s)
        print(f"[train] {name}: states={100 * len(pgrid) ** 2 * nth:,} {s}", flush=True)
        gpi = GPI(cfg)
        # The high-risk-penalty policy needs a few more updates to meet the same
        # value-change tolerance used by the other revised models.
        model_iters = max(num_iters, 30) if name.endswith("_k1_lam10000") else num_iters
        gpi.compute_policy(num_iters=model_iters, tol=tol)
        gpi.save(out)
        print(f"[done] {name}: offline={gpi.offline_sec:.1f}s peak_gpu={gpi.peak_gpu_mb:.0f}MB "
              f"params={gpi.V.n_params():,}", flush=True)
        del gpi
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------
# Controllers for rollouts
# ------------------------------------------------------------------------------------------
_CACHE = {}


def make_controller(spec):
    """spec: ('CEC', kwargs), ('RiskCEC', kwargs) or ('GPI', model_name, online_mode[, overrides])."""
    key = json.dumps(spec, sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    if spec[0] == "CEC":
        from cec import CEC
        ctrl = CEC(**spec[1])
    elif spec[0] == "RiskCEC":
        import torch
        torch.set_num_threads(1)
        from cec_risk import RiskCEC
        kw = dict(spec[1])
        if kw.get("value_model"):
            kw["value_model"] = os.path.join(MODELS_DIR, f"{kw['value_model']}.npz")
        ctrl = RiskCEC(**kw)
    else:
        import torch
        torch.set_num_threads(1)
        from gpi import GPI
        overrides = spec[3] if len(spec) > 3 else {}
        ctrl = GPI.load(os.path.join(MODELS_DIR, f"{spec[1]}.npz"), device="cpu",
                        online_mode=spec[2], **overrides)
    _CACHE[key] = ctrl
    return ctrl


def _risk_sigma(k):
    return max(0.04 * k, 0.01)  # GPI's risk-term sigma: the noise level with a 1 cm floor


def riskcec_spec(k, lam=1000.0, **kw):
    return ("RiskCEC", dict(horizon=10, lam=float(lam), risk_sigma=_risk_sigma(k), **kw))


def hybrid_spec(k, lam=1000.0):
    """Path-aware risk CEC with the path-aware GPI value as terminal cost and the GPI plan as a second start."""
    lam_tag = "" if lam == 1000 else f"_lam{int(lam)}"
    return riskcec_spec(k, lam, hard=True, risk_substeps=4, policy_seed=True,
                        value_model=f"path_grid_medium_{_kname(k)}{lam_tag}")


FOLLOWUP_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0]
HELDOUT_SEEDS = list(range(1000, 1200))


def suite(name, levels=None):
    """List of (label, controller spec, noise scale, seeds)."""
    seeds = list(range(N_SEEDS))
    runs = []
    if name == "riskcec":  # CEC with GPI's calibrated risk (next-sample risk, as in the revised GPI)
        for k in (levels if levels is not None else FOLLOWUP_LEVELS):
            sd = [0] if k == 0 else seeds
            runs += [
                ("RiskCEC-soft", riskcec_spec(k), k, sd),
                ("RiskCEC-hard", riskcec_spec(k, hard=True), k, sd),
                ("RiskCEC-V", riskcec_spec(k, hard=True, value_model=f"revised_grid_medium_{_kname(k)}"), k, sd),
            ]
        if levels is None or 1.0 in levels:
            for n in (20, 40):
                spec = ("RiskCEC", dict(riskcec_spec(1.0, hard=True)[1], horizon=n))
                runs.append((f"RiskCEC-hard-N{n}", spec, 1.0, seeds))
    elif name == "path":  # path-aware risk and constraints
        for k in (levels if levels is not None else FOLLOWUP_LEVELS):
            sd = [0] if k == 0 else seeds
            runs += [
                ("GPI-path", ("GPI", f"path_grid_medium_{_kname(k)}", "lookahead"), k, sd),
                ("CEC-path", ("CEC", {"horizon": 10, "substeps": 4}), k, sd),
                ("Hybrid", hybrid_spec(k), k, sd),
            ]
        if levels is None or 1.0 in levels:
            spec = ("RiskCEC", {key: v for key, v in hybrid_spec(1.0)[1].items() if key != "policy_seed"})
            runs.append(("RiskCEC-path-V", spec, 1.0, seeds))
    elif name == "path_frontier":  # noise x1: lambda sweeps and CEC margins, all path-aware
        for lam in (10, 100, 10000):
            runs.append((f"GPI-path-lam{lam}", ("GPI", f"path_grid_medium_k1_lam{lam}", "lookahead"), 1.0, seeds))
            runs.append((f"Hybrid-lam{lam}", hybrid_spec(1.0, lam), 1.0, seeds))
        for m in (0.025, 0.05, 0.1):
            runs.append((f"CEC-path-m{m:g}", ("CEC", {"horizon": 10, "substeps": 4, "margin": m}), 1.0, seeds))
    elif name == "path_det":  # does the transition noise matter once the risk looks along the path?
        for k in (1.0, 2.0):
            runs.append(("GPI-path-det", ("GPI", f"path_grid_medium_{_kname(k)}_det", "lookahead"), k, seeds))
        spec = ("RiskCEC", dict(hybrid_spec(2.0)[1], warm_start=False))
        runs.append(("Hybrid-planonly", spec, 2.0, seeds))
    elif name == "heldout":  # headline controllers at noise x1 on seeds never used before
        runs += [("CEC", ("CEC", {"horizon": 10}), 1.0, HELDOUT_SEEDS),
                 ("GPI", ("GPI", "revised_grid_medium_k1", "lookahead"), 1.0, HELDOUT_SEEDS),
                 ("GPI-path", ("GPI", "path_grid_medium_k1", "lookahead"), 1.0, HELDOUT_SEEDS),
                 ("Hybrid", hybrid_spec(1.0), 1.0, HELDOUT_SEEDS)]
    elif name == "main":
        for k in (levels if levels is not None else NOISE_LEVELS):
            sd = [0] if k == 0 else seeds  # k = 0 is deterministic
            kn = _kname(k)
            runs += [
                ("CEC", ("CEC", {"horizon": 10}), k, sd),
                ("GPI", ("GPI", f"grid_medium_{kn}", "lookahead"), k, sd),
                ("GPI-lookup", ("GPI", f"grid_medium_{kn}", "lookup"), k, sd),
                ("RBF", ("GPI", f"rbfavg_medium_{kn}", "lookahead"), k, sd),
                ("RBF-LS", ("GPI", f"rbf_medium_{kn}", "lookahead"), k, sd),
                ("GPI-detmodel", ("GPI", "grid_medium_k0", "lookahead"), k, sd),
            ]
    elif name == "revised":
        for k in (levels if levels is not None else NOISE_LEVELS):
            sd = [0] if k == 0 else seeds
            kn = _kname(k)
            runs += [
                ("CEC", ("CEC", {"horizon": 10}), k, sd),
                ("GPI", ("GPI", f"revised_grid_medium_{kn}", "lookahead"), k, sd),
                ("RBF", ("GPI", f"revised_rbfavg_medium_{kn}", "lookahead"), k, sd),
            ]
    elif name == "revised_ablation":
        runs += [("T-noise+risk4cm", ("GPI", "revised_grid_medium_k1", "lookahead"), 1.0, seeds),
                 ("T-noise+risk1cm", ("GPI", "revised_grid_medium_k1_risk1cm", "lookahead"), 1.0, seeds),
                 ("T-det+risk4cm", ("GPI", "revised_grid_medium_k0_risk4cm", "lookahead"), 1.0, seeds),
                 ("T-det+risk1cm", ("GPI", "revised_grid_medium_k0", "lookahead"), 1.0, seeds)]
    elif name == "rbf_evaluator_ablation":
        for k in (0.5, 1.0):
            kn = _kname(k)
            runs.append(("RBF-legacy-kernel", ("GPI", f"rbfavg_medium_{kn}", "lookahead"), k, seeds))
            runs.append(("RBF-legacy-interpolated",
                         ("GPI", f"rbfavg_medium_{kn}", "lookahead",
                          {"rbf_online_value": "interpolated"}), k, seeds))
    elif name == "revised_lambda":
        runs.append(("GPI-lam1000", ("GPI", "revised_grid_medium_k1", "lookahead"), 1.0, seeds))
        for lam in (10, 100, 10000):
            runs.append((f"GPI-lam{lam}", ("GPI", f"revised_grid_medium_k1_lam{lam}", "lookahead"), 1.0, seeds))
    elif name == "grid":
        for g in ("coarse", "medium", "fine"):
            runs.append((f"GPI-{g}", ("GPI", f"grid_{g}_k1", "lookahead"), 1.0, seeds))
            runs.append((f"GPI-{g}-lookup", ("GPI", f"grid_{g}_k1", "lookup"), 1.0, seeds))
    elif name == "rbf":
        runs.append(("RBF-smooth", ("GPI", "rbfavg_smooth_k1", "lookahead"), 1.0, seeds))
        runs.append(("RBF-LS-smooth", ("GPI", "rbf_smooth_k1", "lookahead"), 1.0, seeds))
    elif name == "pilot_margin":  # pilot A: CEC with tightened obstacle constraints
        for k in (0.25, 1.0):
            for m in (0.0, 0.025, 0.05, 0.1):
                runs.append((f"CEC-m{m:g}", ("CEC", {"horizon": 10, "margin": m}), k, seeds))
    elif name == "ablation2x2":  # which part of the noise model buys GPI's safety? (sigma x 1)
        runs += [("T-noise+risk4cm", ("GPI", "grid_medium_k1", "lookahead"), 1.0, seeds),
                 ("T-noise+risk1cm", ("GPI", "grid_medium_k1_risk1cm", "lookahead"), 1.0, seeds),
                 ("T-det+risk4cm", ("GPI", "grid_medium_k0_risk4cm", "lookahead"), 1.0, seeds),
                 ("T-det+risk1cm", ("GPI", "grid_medium_k0", "lookahead"), 1.0, seeds)]
    elif name == "pilot_lambda":  # pilot B: GPI collision-penalty sweep at sigma x 1
        runs.append(("GPI-lam1000", ("GPI", "grid_medium_k1", "lookahead"), 1.0, seeds))
        for lam in (10, 100, 10000):
            runs.append((f"GPI-lam{lam}", ("GPI", f"grid_medium_k1_lam{lam}", "lookahead"), 1.0, seeds))
    else:
        raise ValueError(name)
    return runs


def _work(task):
    label, spec, k, seed = task
    import torch
    torch.set_num_threads(1)
    ctrl = make_controller(spec)
    r = C.rollout(ctrl, noise_scale=k, seed=seed, keep_traj=True)
    r["controller"] = label
    r["spec"] = json.dumps(spec)
    return r


def run_suite(name, workers=28, levels=None, tag=""):
    tasks = [(lab, spec, k, s) for lab, spec, k, seeds in suite(name, levels) for s in seeds]
    print(f"[rollout] suite={name} episodes={len(tasks)} workers={workers}", flush=True)
    t0 = perf_counter()
    rows = []
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=4)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)}  {perf_counter() - t0:.0f}s", flush=True)
    with open(os.path.join(RESULTS, f"rollouts_{name}{tag}.pkl"), "wb") as fh:
        pickle.dump(rows, fh)
    print(f"[rollout] done in {perf_counter() - t0:.0f}s", flush=True)


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def timing(n_eps=3, revised=False):
    """Online per-step control time measured in a single process (no CPU contention)."""
    import torch
    torch.set_num_threads(1)
    if revised:
        specs = [
            ("CEC N=10", ("CEC", {"horizon": 10})),
            ("GPI lookahead", ("GPI", "revised_grid_medium_k1", "lookahead")),
            ("GPI lookup", ("GPI", "revised_grid_medium_k1", "lookup")),
            ("RBF lookahead", ("GPI", "revised_rbfavg_medium_k1", "lookahead")),
        ]
    else:
        specs = [
            ("CEC N=10", ("CEC", {"horizon": 10})),
            ("CEC N=20", ("CEC", {"horizon": 20})),
            ("GPI lookahead (medium)", ("GPI", "grid_medium_k1", "lookahead")),
            ("GPI lookup (medium)", ("GPI", "grid_medium_k1", "lookup")),
            ("RBF lookahead (exact kernel)", ("GPI", "rbfavg_medium_k1", "lookahead")),
            ("GPI lookahead (fine)", ("GPI", "grid_fine_k1", "lookahead")),
        ]
    out = []
    for label, spec in specs:
        if spec[0] == "GPI" and not os.path.exists(os.path.join(MODELS_DIR, f"{spec[1]}.npz")):
            print(f"[skip] {label}: model missing", flush=True)
            continue
        ctrl = make_controller(spec)
        ts = []
        for s in range(n_eps):
            r = C.rollout(ctrl, noise_scale=1.0, seed=1000 + s, keep_traj=False)
            ts.append((r["ctrl_ms_mean"], r["ctrl_ms_p95"], r["ctrl_ms_max"]))
        ts = np.array(ts)
        out.append(dict(controller=label, ms_mean=ts[:, 0].mean(), ms_p95=ts[:, 1].mean(), ms_max=ts[:, 2].max()))
        print(out[-1], flush=True)
    with open(os.path.join(RESULTS, "revised_timing.json" if revised else "timing.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def timing_followup(n_eps=3):
    """Per-step control time of the follow-up controllers at noise x1, single process."""
    import torch
    torch.set_num_threads(1)
    specs = [
        ("CEC, path constraints", ("CEC", {"horizon": 10, "substeps": 4})),
        ("risk-aware CEC, constraints", riskcec_spec(1.0, hard=True)),
        ("risk-aware CEC + GPI value", riskcec_spec(1.0, hard=True, value_model="revised_grid_medium_k1")),
        ("GPI, path risk, lookahead", ("GPI", "path_grid_medium_k1", "lookahead")),
        ("hybrid (path risk, value, policy start)", hybrid_spec(1.0)),
    ]
    out = []
    for label, spec in specs:
        ctrl = make_controller(spec)
        ts = []
        for s in range(n_eps):
            r = C.rollout(ctrl, noise_scale=1.0, seed=1000 + s, keep_traj=False)
            ts.append((r["ctrl_ms_mean"], r["ctrl_ms_p95"], r["ctrl_ms_max"]))
        ts = np.array(ts)
        out.append(dict(controller=label, ms_mean=ts[:, 0].mean(), ms_p95=ts[:, 1].mean(), ms_max=ts[:, 2].max()))
        print(out[-1], flush=True)
    with open(os.path.join(RESULTS, "followup_timing.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def offline_timing(n_iters=2):
    """Clean (single job) offline cost of one GPI iteration (1 improvement + num_evals evaluations)."""
    import torch
    from gpi import GPI, GpiConfig, GRID_PRESETS, theta_grid
    out = []
    for name, grid, extra in (("coarse", "coarse", {}), ("medium", "medium", {}), ("fine", "fine", {}),
                              ("medium-RBF(avg)", "medium", dict(value_type="rbf", rbf_fit="avg"))):
        pgrid, nth = GRID_PRESETS[grid]
        g = GPI(GpiConfig(ex_space=pgrid, ey_space=pgrid, eth_space=theta_grid(nth), device="cuda",
                          noise_scale=1.0, **extra))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = perf_counter()
        g.compute_policy(num_iters=n_iters, tol=-1, verbose=False)
        torch.cuda.synchronize()
        sec = (perf_counter() - t0) / n_iters
        rec = dict(grid=name, n_states=g.T * g.S, n_actions=g.nA, n_noise_nodes=g.nW,
                   n_params=g.V.n_params(), value_MB=4 * g.V.n_params() / 2 ** 20,
                   sec_per_iter=sec, peak_gpu_MB=torch.cuda.max_memory_allocated() / 2 ** 20,
                   dense_P_table_GB=g.T * g.S * g.nA * 8 * 4 * 4 / 2 ** 30)  # (.., 8 nbrs, 4 fields) fp32
        print(rec, flush=True)
        out.append(rec)
        del g
        torch.cuda.empty_cache()
    with open(os.path.join(RESULTS, "offline_timing.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "rollout", "timing", "timing_revised", "timing_followup",
                                    "offline_timing", "list"])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--suite", default="main")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--levels", type=float, nargs="*", default=None, help="subset of noise levels (main suite)")
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _part1")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.cmd == "train":
        train(a.models)
    elif a.cmd == "rollout":
        run_suite(a.suite, a.workers, a.levels, a.tag)
    elif a.cmd == "timing":
        timing()
    elif a.cmd == "timing_followup":
        timing_followup()
    elif a.cmd == "timing_revised":
        timing(revised=True)
    elif a.cmd == "offline_timing":
        offline_timing()
    else:
        print("\n".join(model_specs()))
