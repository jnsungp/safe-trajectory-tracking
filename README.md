# Safe Trajectory Tracking under Motion Uncertainty

**Blog post:** https://jnsungp.github.io/blog/safe-trajectory-tracking/ · **Aggregate results:** [`starter_code/results/revised_summary.json`](starter_code/results/revised_summary.json) (main experiment), [`starter_code/results/followup_summary.json`](starter_code/results/followup_summary.json) (follow-up)

This repository continues my project from [UCSD ECE 276B: Planning & Learning in Robotics](https://natanaso.github.io/ece276b/) (Project 3). A differential-drive robot tracks a figure-eight reference around four circular obstacles under Gaussian motion noise. The experiment compares three controllers that share the simulator but use different objectives and noise information:

| method | idea |
|---|---|
| **CEC** | ten-step CasADi/IPOPT NLP with the noise set to zero, re-solved every step |
| **Tabular GPI** | 3.36 M-state discretized MDP solved on the GPU; exact 2-D Gaussian collision probability for each inflated obstacle |
| **RBF GPI** | the same backups with a 450 k-weight Gaussian-RBF value function (normalized-kernel averager) |

A follow-up adds two controllers:

| method | idea |
|---|---|
| **Risk-aware CEC** | CEC with GPI's stage cost, including the calibrated collision probability; optionally the nominal obstacle constraints and GPI's value function as the terminal cost |
| **Hybrid** | risk-aware CEC with a path-aware risk, GPI's value as the terminal cost, and GPI's ten-step plan as a second initial guess |

## Main results (200 paired seeds per noise level)

- The inflated obstacles leave a **5.1 cm gap**, and the reference passes *inside* two of them. The results are about how each controller handles that gap.
- **CEC** tracks most closely (9.6 cm at noise ×1) but collides in 100% of noisy episodes, on average after 24 steps: its optimum sits on an active constraint.
- **Tabular GPI** collides at a sampled position in 0%, 5.5%, 41.5% and 20.5% of episodes at noise ×0.25, ×0.5, ×1 and ×2. At high noise it detours around the gap, at the cost of tracking error. (The follow-up shows that part of this margin exists only at the samples.)
- A 2×2 ablation at noise ×1 shows that GPI's margin comes from the **calibrated collision probability**, not from the Gaussian transition in the Bellman backup (41.5% vs 38.5% collisions with and without transition noise, p = 0.53).
- **RBF GPI** stores 7.5× fewer parameters and collides far more often (44.5% vs 5.5% at ×0.5). A plain least-squares RBF fit converges to a value function that is negative on 76% of states, even though every cost is non-negative.
- The ordering holds from four different starting phases of the reference.

## Follow-up: risk-aware CEC and collisions between samples

- **Giving CEC GPI's collision probability made it worse.** With the same penalty (λ = 1000) and noise level, it collided in 100% of episodes at noise ×0.25, because the probability is flat outside a thin band around each obstacle and predicted positions 17 cm apart skipped that band. At ×1 it stalled within its ten-step window (100%, 79 cm tracking error).
- **Two fixes.** The nominal obstacle constraints give the risk a gradient (0.5% at ×0.25). GPI's value function as the terminal cost removes the stall (35 cm at ×1), but the controller still collides more often than GPI (54.5% vs 5.5% at ×0.5, 78% vs 41.5% at ×1).
- **The samples hide collisions.** GPI crosses the gaps at full speed so that its two samples land on either side of the 5 cm gap. Checked along the path between samples, it collides in 72.5% of episodes at ×0.5 (not 5.5%) and 84% at ×1 (not 41.5%).
- **A path-aware risk** takes the largest collision probability at four points of each step. GPI trained with it detours around both gaps at ×1 and collides along the path in 0% of episodes, at 44 cm tracking error.
- **The hybrid** is safer than path-aware GPI at ×0.25 and ×0.5, nearly as safe at ×1 (4% vs 0%; 2% vs 0% on held-out seeds), and tracks closer at every noise level (35 vs 44 cm at ×1). At ×2 its noise-free predictions cost it safety (72% vs 20.5%); GPI with noise-free transitions shows the same drop (42% vs 20.5%).

## Code

| file | purpose |
|---|---|
| [`common.py`](starter_code/common.py) | dynamics, cost, geometry, paired-noise rollout, first-collision metrics |
| [`cec.py`](starter_code/cec.py) | multiple-shooting CEC with IPOPT; optional constraints along each step's arc |
| [`cec_risk.py`](starter_code/cec_risk.py) | risk-aware CEC and the hybrid: GPI's risk penalty, GPI's value as a local-spline terminal cost, path-aware risk, GPI plan as a start |
| [`gpi.py`](starter_code/gpi.py) | GPU GPI: on-the-fly transitions, collision-risk term (next sample or along the arc), online control |
| [`value_function.py`](starter_code/value_function.py) | tabular and RBF value functions |
| [`experiments.py`](starter_code/experiments.py) | model specifications, training, rollout suites, timing |
| [`phase_robustness.py`](starter_code/phase_robustness.py) | two-period runs from four aligned starting phases |
| [`revised_analysis.py`](starter_code/revised_analysis.py), [`followup_analysis.py`](starter_code/followup_analysis.py) | aggregate statistics and paired tests; collisions along the path |
| [`blog_figs.py`](starter_code/blog_figs.py), [`followup_figs.py`](starter_code/followup_figs.py) | figures and GIF for the blog post |

## Reproduce

```bash
conda env create -f environment.yml && conda activate traj
cd starter_code
python test_revised_risk.py                                                        # checks of the collision-risk term
python experiments.py train --models $(python experiments.py list | rg '^revised_')  # CUDA GPU, a few minutes per model
python experiments.py rollout --suite revised
python experiments.py rollout --suite revised_ablation
python experiments.py rollout --suite revised_lambda
python phase_robustness.py
python experiments.py timing_revised
python revised_analysis.py && python blog_figs.py
python main.py --controller cec                                                    # single-episode demo (also gpi / rbf / p)

# follow-up (needs the rollouts above)
python test_followup.py
python experiments.py train --models $(python experiments.py list | rg '^path_')
python experiments.py rollout --suite riskcec
python experiments.py rollout --suite path
python experiments.py rollout --suite path_frontier
python experiments.py rollout --suite path_det
python experiments.py rollout --suite heldout
python experiments.py timing_followup
python followup_analysis.py && python followup_figs.py
```

Trained models (`results/models/`) and raw rollouts (`results/*.pkl`) are not committed; the commands above regenerate them. `report.md` holds my earlier Korean notes on this project from a first round of experiments with a simpler risk approximation; the blog post and the numbers above come from `results/revised_summary.json` and `results/followup_summary.json`.

## Acknowledgments

Built on the ECE 276B starter code (Nikolay Atanasov, UC San Diego). The implementation and experiments were done with the help of AI coding assistants (Claude Code and Codex).
