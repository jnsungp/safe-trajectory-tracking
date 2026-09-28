# Code: CEC, tabular GPI and RBF GPI for PR3

This folder builds on the ECE 276B PR3 starter code. The main entry points are:

## `main.py` (single closed-loop episode)

```bash
python main.py --controller cec                     # receding-horizon CEC
python main.py --controller gpi --model results/models/grid_medium_k1.npz
python main.py --controller rbf --model results/models/rbfavg_medium_k1.npz
python main.py --controller p                       # starter P controller (baseline)
# options: --noise K (noise scale, default 1), --seed S, --horizon N (CEC), --show (animate), --mujoco
```

It prints tracking errors in the starter's format, the mean control time, and the collision statistics.

## Files

| file | contents |
|---|---|
| `common.py` | exact-discretization dynamics (eq. 1), error dynamics (eq. 2), stage cost (eq. 3), obstacles and clearance, `rollout()` harness with common random numbers and metrics |
| `cec.py` | `CEC`: parametric multiple-shooting NLP (CasADi/IPOPT) with slack-penalized obstacle constraints and warm start |
| `gpi.py` | `GpiConfig`, `GPI`: GPU modified policy iteration on the discretized MDP (Gauss–Hermite noise quadrature, multilinear interpolation, collision-risk stage cost), online one-step lookahead or table lookup, save/load |
| `value_function.py` | `GridValueFunction` (tabular) and `FeatureValueFunction` (Gaussian RBF with an exact Kronecker ridge LS fit, or a normalized-kernel averager) |
| `experiments.py` | model zoo (`train`), rollout suites (`rollout --suite main/grid/rbf/pilot_margin/pilot_lambda/ablation2x2`), `timing`, `offline_timing` |
| `analyze.py`, `extra_stats.py` | statistics (Wilson CIs, paired Wilcoxon, exact McNemar) and figures in `results/figs/` |
| `make_media.py` | side-by-side rollout GIFs |
| `utils.py`, `mujoco_car.py`, `mujoco_assets/` | starter code: reference trajectory, visualization, MuJoCo car |

Trained GPI/RBF models are written to `results/models/*.npz` by `python experiments.py train --models ...`. That step needs a CUDA GPU; the online controllers run on CPU.

## Revised results-first study

The revised experiment uses the exact 2-D Gaussian probability of entering each inflated circular obstacle. Because the four inflated disks are disjoint, their probabilities add. A 0.5 mm distance lookup keeps this calculation fast on the GPU; combining obstacle and workspace events remains approximate near the outer boundary. RBF online lookahead uses the same rendered-value interpolation as its offline Bellman backup. The old model names and `report.md` remain available as the original experiment record.

```bash
python test_revised_risk.py
python experiments.py train --models $(python experiments.py list | rg '^revised_')
python experiments.py rollout --suite revised
python experiments.py rollout --suite revised_ablation
python experiments.py rollout --suite revised_lambda
python experiments.py rollout --suite rbf_evaluator_ablation
python phase_robustness.py
python experiments.py timing_revised
python revised_analysis.py
python revised_figs.py
```

The main 200-seed rollouts are in `results/rollouts_revised*.pkl` (ignored by Git); the reviewable aggregate is `results/revised_summary.json`. The phase experiment aligns the initial state with the reference at phases 0, 25, 50 and 75, and runs exactly two periods from each phase.
