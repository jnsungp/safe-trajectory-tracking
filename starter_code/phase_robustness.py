"""Two complete reference periods from four aligned starting phases at noise x1.

This removes the default x0=reference(2), t0=0 mismatch and the 2.4-period exposure.
"""
import json
import os
import pickle
from multiprocessing import get_context

import numpy as np

import analyze as A
import common as C
from experiments import make_controller


SPECS = {
    "CEC": ("CEC", {"horizon": 10}),
    "GPI": ("GPI", "revised_grid_medium_k1", "lookahead"),
    "RBF": ("GPI", "revised_rbfavg_medium_k1", "lookahead"),
}
PHASES = (0, 25, 50, 75)


def work(task):
    import torch
    torch.set_num_threads(1)
    name, phase, seed = task
    controller = make_controller(SPECS[name])
    rec = C.rollout(controller, noise_scale=1.0, seed=seed, n_steps=200,
                    x0=C.ref(phase), start_t=phase)
    rec["controller"] = name
    rec["phase"] = phase
    return rec


def main(workers=12):
    tasks = [(name, phase, seed) for phase in PHASES for name in SPECS for seed in range(200)]
    with get_context("spawn").Pool(workers) as pool:
        rows = list(pool.imap_unordered(work, tasks, chunksize=4))
    with open(os.path.join(A.RES, "rollouts_revised_phase.pkl"), "wb") as fh:
        pickle.dump(rows, fh)
    summary = []
    for phase in PHASES:
        for name in SPECS:
            rr = [r for r in rows if r["phase"] == phase and r["controller"] == name]
            n_hit = sum(r["collided"] for r in rr)
            summary.append(dict(phase=phase, controller=name, n=len(rr), p_hit=n_hit / len(rr),
                                p_hit_ci=A.wilson(n_hit, len(rr)),
                                safe_steps_mean=float(np.mean([r["safe_steps_until_hit"] for r in rr])),
                                pos_err=float(np.mean([r["mean_pos_err"] for r in rr])),
                                stage_cost=float(np.mean([r["mean_stage_cost"] for r in rr]))))
    with open(os.path.join(A.RES, "revised_phase.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
