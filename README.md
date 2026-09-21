# DQN-family satellite-ground scheduling reproducibility package

This repository contains the code, locked model checkpoints, raw results, and validation records for *Statistical Evidence for a DQN-Family Advantage in a Resource-Coupled Satellite-Ground Scheduling Benchmark*. The evidence applies to the synthetic simulator with clipped, soft-constrained resources described in the manuscript.

## Included materials

- `dqn_family_satellite_ground/`: simulator, DQN agents, planning baselines, experiment and reporting programs, MATLAB plotting source, and unit tests.
- `dqn_family_satellite_ground/results/reviewer_revision_state_complete/`: the locked original experiments, including 180 checkpoints and their 180 training curves.
- `dqn_family_satellite_ground/results/revision_round2/`: discount-factor sensitivity, 300/600/900-episode checkpoints, eight-step greedy rollout, matched CPU timing, seed-level evaluations, bootstrap/Holm inference, and validation records. This directory contains 840 additional checkpoints and 840 corresponding curves.
- `dqn_family_satellite_ground_manuscript.pdf`: the current clean manuscript.
- `MANIFEST.tsv`, `SHA256SUMS.txt`, and `verify_package.py`: full package inventory, hashes, and structural validation.

The full package therefore contains 1,020 checkpoints and 1,020 training curves. All formal learned-policy experiments use model seeds 800--819. Each reported held-out cell averages 20 shared traces per model-seed block.

## Algorithm identifiers

Machine-readable identifiers follow the manuscript terminology:

- `standard_dqn`
- `full_action_q_dqn`
- `immediate_advantage_dqn`
- `centered_full_action_dqn`
- `double_dqn`
- `double_centered_full_action_dqn`
- `contextual_bandit`

The new planning baselines are `mpc4` and `greedy8`.

## Environment

The recorded runs used Python 3.13.9, NumPy 2.4.1, pandas 3.0.3, PyTorch 2.8.0 with CUDA 12.9, and Matplotlib 3.10.8 on Windows 11. Install the direct dependencies with:

```powershell
python -m pip install -r dqn_family_satellite_ground/requirements.txt
```

## Verify the complete package

Run from the repository root:

```powershell
python verify_package.py
python -m unittest discover -s dqn_family_satellite_ground/tests -v
python -m dqn_family_satellite_ground.reviewer_experiments validate --seeds 800-819
python -m dqn_family_satellite_ground.revision_experiments validate --results dqn_family_satellite_ground/results/revision_round2 --seeds 800-819
```

The package verifier excludes `.git` from the inventory, checks every distributed SHA-256 digest, rejects manuscript source assets, validates public algorithm identifiers, and independently checks original and second-round checkpoint counts and training budgets.

## Reproduce the original result chain

```powershell
python -m dqn_family_satellite_ground.reviewer_experiments all --seeds 800-819 --resume
python -m dqn_family_satellite_ground.extended_revision --seeds 800-819 --stage all --resume
python -m dqn_family_satellite_ground.transition_audit
python -m dqn_family_satellite_ground.reviewer_reporting
python -m dqn_family_satellite_ground.extended_reporting
python -m dqn_family_satellite_ground.plot_dqn_family_summary
matlab -batch "addpath(fullfile(pwd,'dqn_family_satellite_ground')); plot_reviewer_results_matlab(pwd)"
```

## Reproduce the second-round experiments

The following command reuses every validated bundled checkpoint and regenerates evaluations, inference, and validation records:

```powershell
python -m dqn_family_satellite_ground.revision_experiments all --results dqn_family_satellite_ground/results/revision_round2 --seeds 800-819 --episodes 600 --horizon 64 --traces 20 --resume
```

To regenerate the two manuscript figures, runtime table, and audit report in a separate output directory:

```powershell
python -m dqn_family_satellite_ground.revision_reporting --results dqn_family_satellite_ground/results/revision_round2 --manuscript-dir reproduced_outputs/revision_round2
```

For a from-scratch run that preserves the distributed results, replace the `--results` argument with a new path and omit `--resume`.

## Statistical design

The second-round sensitivity experiment uses six objectives, four discount factors (`0.90`, `0.95`, `0.97`, and `0.99`), 20 model seeds, and 20 paired held-out traces per seed. Training-length checkpoints are evaluated at 300, 600, and 900 episodes with epsilon decay fixed at 13,440 interactions. Bootstrap intervals use 10,000 model-seed block resamples. Gamma, training-length, DQN-versus-MPC, and DQN-versus-greedy comparisons use separate Holm families.

## Repository

Permanent repository: <https://github.com/noob32123/dqn_family_satellite_ground_reproducibility>
