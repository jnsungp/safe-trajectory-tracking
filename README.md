# Safe Trajectory Tracking under Motion Uncertainty

**Blog post:** https://jnsungp.github.io/blog/safe-trajectory-tracking/ · **Aggregate results:** [`starter_code/results/revised_summary.json`](starter_code/results/revised_summary.json)

This repository continues my project from [UCSD ECE 276B: Planning & Learning in Robotics](https://natanaso.github.io/ece276b/) (Project 3). A differential-drive robot tracks a figure-eight reference around four circular obstacles under Gaussian motion noise. The experiment compares three controllers that share the simulator but use different objectives and noise information:

| method | idea |
|---|---|
| **CEC** | ten-step CasADi/IPOPT NLP with the noise set to zero, re-solved every step |
| **Tabular GPI** | 3.36 M-state discretized MDP solved on the GPU; exact 2-D Gaussian collision probability for each inflated obstacle |
| **RBF GPI** | the same backups with a 450 k-weight Gaussian-RBF value function (normalized-kernel averager) |

## Main results (200 paired seeds per noise level)

- The inflated obstacles leave a **5.1 cm gap**, and the reference passes *inside* two of them. The results are about how each controller handles that gap.
- **CEC** tracks most closely (9.6 cm at noise ×1) but collides in 100% of noisy episodes, on average after 24 steps: its optimum sits on an active constraint.
- **Tabular GPI** collides in 0%, 5.5%, 41.5% and 20.5% of episodes at noise ×0.25, ×0.5, ×1 and ×2. At high noise it detours around the gap, at the cost of tracking error.
- A 2×2 ablation at noise ×1 shows that GPI's margin comes from the **calibrated collision probability**, not from the Gaussian transition in the Bellman backup (41.5% vs 38.5% collisions with and without transition noise, p = 0.53).
- **RBF GPI** stores 7.5× fewer parameters and collides far more often (44.5% vs 5.5% at ×0.5). A plain least-squares RBF fit converges to a value function that is negative on 76% of states, even though every cost is non-negative.
- The ordering holds from four different starting phases of the reference.

## Code

| file | purpose |
|---|---|
| [`common.py`](starter_code/common.py) | dynamics, cost, geometry, paired-noise rollout, first-collision metrics |
| [`cec.py`](starter_code/cec.py) | multiple-shooting CEC with IPOPT |
| [`gpi.py`](starter_code/gpi.py) | GPU GPI: on-the-fly transitions, collision-risk term, online control |
| [`value_function.py`](starter_code/value_function.py) | tabular and RBF value functions |
| [`experiments.py`](starter_code/experiments.py) | model specifications, training, rollout suites, timing |
| [`phase_robustness.py`](starter_code/phase_robustness.py) | two-period runs from four aligned starting phases |
| [`revised_analysis.py`](starter_code/revised_analysis.py) | aggregate statistics and paired tests |
| [`blog_figs.py`](starter_code/blog_figs.py) | figures and GIF for the blog post |

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
```

Trained models (`results/models/`) and raw rollouts (`results/*.pkl`) are not committed; the commands above regenerate them. `report.md` holds my earlier Korean notes on this project from a first round of experiments with a simpler risk approximation; the blog post and the numbers above come from `results/revised_summary.json`.

## Acknowledgments

Built on the ECE 276B starter code (Nikolay Atanasov, UC San Diego). The implementation and experiments were done with the help of AI coding assistants (Claude Code and Codex).
