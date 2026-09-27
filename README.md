# Safe Trajectory Tracking under Motion Uncertainty

**Blog post:** https://jnsungp.github.io/blog/safe-trajectory-tracking/ · **Technical report (Korean):** [`report.md`](report.md)

This repository continues my project from [UCSD ECE 276B: Planning & Learning in Robotics](https://natanaso.github.io/ece276b/), Project 3 (infinite-horizon stochastic optimal control). The robot is a differential drive that has to track a figure-eight reference while avoiding four circular obstacles, with Gaussian motion noise at every step. The repo compares three approximations of the same stochastic optimal controller:

| method | file | idea |
|---|---|---|
| Receding-horizon **CEC** | `starter_code/cec.py` | 10-step CasADi/IPOPT NLP, noise set to zero, re-solved every step (~5 ms) |
| Tabular **GPI** | `starter_code/gpi.py`, `value_function.py` | 3.36 M-state discretized MDP; Gauss–Hermite noise quadrature plus multilinear interpolation; transitions computed on the fly on the GPU (<1 GB instead of a 44 GB table) |
| **RBF GPI** | `starter_code/value_function.py` | Gaussian RBF value function (7.5× fewer parameters). Exact Kronecker least-squares fit, plus a normalized-kernel *averager* that keeps projected GPI a contraction |

I stated five hypotheses before running anything, then tested them on 200 paired noise seeds (common random numbers) at five noise levels.

## Main findings

- **A 5 cm trap.** The reference passes *inside* the inflated obstacles C1/C2, and the gap between inflated C1 and C3 (and between C2 and C4) is only 5 cm. The results separate into *where inside the corridor* a controller sits and *whether it takes the corridor at all*.
- **H1 (supported, with a catch).** Without noise, CEC tracks best (0.047 m vs 0.065 m for GPI). It does so by riding the constraint (clearance 0.000), and the continuous path of eq. (1) cuts 1.7 cm into the obstacles between samples.
- **H2 (supported, and stronger).** CEC's safety breaks at once: 100% of episodes collide at the *smallest* noise tested (σ×0.25). Its optimum sits on an active constraint, so noise only changes the penetration depth.
- **H3 (wrong as stated).** GPI's tracking-cost gap to CEC *widens* with noise (discounted cost 1.2× → 8.7×), but it collides far less (0% vs 100% at σ×0.25; 37% vs 100% at σ×1). A 2×2 ablation shows the safety comes from the **noise-calibrated collision-risk term**, *not* from the Gaussian transition in the Bellman backup (37% vs 41% collisions with/without transition noise, p = 0.39).
- **H4 (supported, matters less than expected).** Online time is 1.1 ms (lookahead) or 0.14 ms (table lookup) vs 5.3 ms for CEC; offline cost is nearly linear in the number of states. Pure lookup picks the same route but loses the in-corridor margin (92% vs 0% collisions at σ×0.25).
- **H5 (supported).** The averager-RBF is less safe at every non-zero noise level, and its blur is global. Plain least-squares RBF converges to a fixed point that is negative on 76% of states even though V ≥ 0: a non-contractive projection amplified by the greedy step.

## Layout

```
report.md                    detailed report (Korean), every number traced to results/
environment.yml              conda env "traj"
starter_code/
  common.py                  dynamics (eq. 1–2), cost, geometry, rollout harness (common random numbers)
  cec.py                     receding-horizon CEC (CasADi / IPOPT)
  gpi.py                     GPU GPI: on-the-fly transitions, risk term, modified policy iteration, online control
  value_function.py          tabular and RBF (least-squares / averager) value functions
  experiments.py             train / rollout suites / timing
  analyze.py                 statistics and figures      extra_stats.py  supporting numbers
  make_media.py              GIFs                        main.py         single-episode demo
  utils.py, mujoco_car.py    course starter code (visualization, MuJoCo car)
  results/                   summary.json, extra_stats.json, timing*.json, logs/, figs/
```

## Reproduce

```bash
conda env create -f environment.yml && conda activate traj
cd starter_code
python main.py --controller cec                      # quick demo (also: gpi / rbf / p)
python experiments.py list                           # all model configs
python experiments.py train --models grid_medium_k1 rbfavg_medium_k1   # needs a CUDA GPU (~7 min each on an L40S)
python experiments.py rollout --suite main           # also: grid, rbf, pilot_margin, pilot_lambda, ablation2x2
python experiments.py timing && python experiments.py offline_timing
python analyze.py && python extra_stats.py && python make_media.py
```

Trained models (`results/models/`, 174 MB) and raw rollouts (`results/*.pkl`) are not committed; the commands above regenerate them.

## Acknowledgments

Built on the ECE 276B starter code (Nikolay Atanasov, UC San Diego). The implementation, experiments and write-up were done with the help of an AI coding assistant (Claude Code). The code was checked by independent review passes, and every reported number was recomputed from the raw rollouts.
