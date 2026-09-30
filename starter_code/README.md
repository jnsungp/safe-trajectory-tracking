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
| `cec.py` | `CEC`: parametric multiple-shooting NLP (CasADi/IPOPT) with slack-penalized obstacle constraints and warm start; `substeps` also constrains points along each step's arc |
| `cec_risk.py` | `RiskCEC`: the same NLP with GPI's collision-probability penalty, optional obstacle constraints, GPI's value function as the terminal cost (`LocalValue`), a path-aware risk, and GPI's plan as a second initial guess |
| `gpi.py` | `GpiConfig`, `GPI`: GPU modified policy iteration on the discretized MDP (Gauss–Hermite noise quadrature, multilinear interpolation, collision-risk stage cost; `risk_substeps` scores the whole step's arc), online one-step lookahead or table lookup, save/load |
| `value_function.py` | `GridValueFunction` (tabular) and `FeatureValueFunction` (Gaussian RBF with an exact Kronecker ridge LS fit, or a normalized-kernel averager) |
| `experiments.py` | model zoo (`train`), rollout suites (`rollout --suite revised/riskcec/path/path_frontier/heldout/...`), `timing`, `timing_revised`, `timing_followup`, `offline_timing` |
| `analyze.py`, `extra_stats.py` | statistics (Wilson CIs, paired Wilcoxon, exact McNemar) and figures in `results/figs/` |
| `blog_figs.py`, `followup_figs.py` | figures and GIF for the blog post |
| `revised_analysis.py`, `followup_analysis.py` | aggregate statistics for the main and follow-up experiments (`results/revised_summary.json`, `results/followup_summary.json`) |
| `make_media.py` | side-by-side rollout GIFs (first round of experiments) |
| `utils.py`, `mujoco_car.py`, `mujoco_assets/` | starter code: reference trajectory, visualization, MuJoCo car |

Trained GPI/RBF models are written to `results/models/*.npz` by `python experiments.py train --models ...`. That step needs a CUDA GPU; the online controllers run on CPU.

## Main experiment

GPI's risk term uses the exact 2-D Gaussian probability of entering each inflated circular obstacle. Because the four inflated disks do not overlap, their probabilities add; a 0.5 mm distance lookup keeps the calculation fast on the GPU. The union with the workspace boundary is approximate where an obstacle extends past the square. RBF online lookahead evaluates the value function exactly as the offline Bellman backup does. Models whose names start with `revised_` belong to this experiment.

```bash
python test_revised_risk.py
python experiments.py train --models $(python experiments.py list | rg '^revised_')
python experiments.py rollout --suite revised
python experiments.py rollout --suite revised_ablation
python experiments.py rollout --suite revised_lambda
python phase_robustness.py
python experiments.py timing_revised
python revised_analysis.py
python blog_figs.py
```

The 200-seed rollouts are written to `results/rollouts_revised*.pkl` (not committed); the aggregate is `results/revised_summary.json`. The phase experiment starts the robot on the reference at phases 0, 25, 50 and 75 and runs exactly two periods from each.

## Follow-up experiment

`RiskCEC` gives CEC the collision probability that GPI minimizes. The follow-up also scores collisions along the path between samples: the noise-free arc of each step with that step's noise blended in linearly, at 21 points (`analyze.arc_clearance`, `followup_analysis.path_clearance_per_step`). Models whose names start with `path_` use a risk term that takes the largest collision probability at four points along each step's arc (`risk_substeps=4`).

```bash
python test_followup.py
python experiments.py train --models $(python experiments.py list | rg '^path_')
python experiments.py rollout --suite riskcec        # risk-aware CEC variants, next-sample risk
python experiments.py rollout --suite path           # path-aware GPI, CEC with arc constraints, hybrid
python experiments.py rollout --suite path_frontier  # lambda sweeps and CEC margins at noise x1
python experiments.py rollout --suite path_det       # GPI with noise-free transitions and the path risk
python experiments.py rollout --suite heldout        # seeds 1000-1199 at noise x1
python experiments.py timing_followup
python followup_analysis.py && python followup_figs.py
```

The rollouts are written to `results/rollouts_{riskcec,path,path_frontier,path_det,heldout}.pkl` (not committed); the aggregate is `results/followup_summary.json`.
