# DQN-family satellite-ground scheduling reproducibility package

This package is synchronized with the current manuscript PDF bundled at the package root. It contains only the materials needed to reproduce the experiments reported in that manuscript.

## Included

- `dqn_family_satellite_ground/`: current Python experiment, validation, reporting, and figure-generation source; the MATLAB figure-generation script; unit tests; and pinned direct dependencies.
- `dqn_family_satellite_ground/results/reviewer_revision_state_complete/`: locked raw evaluations, seed-level summaries, inference files, metadata, validation records, 180 trained checkpoints, and 180 corresponding training curves.
- `dqn_family_satellite_ground_manuscript.pdf`: the current English manuscript.
- `MANIFEST.tsv`, `SHA256SUMS.txt`, and `verify_package.py`: package inventory and integrity verification.

The 180 checkpoints comprise 120 checkpoints for the six reported DQN objectives, 20 contextual-bandit checkpoints, and 40 centered-full-action checkpoints used only for the two preview-mismatch conditions. Model seeds are 800--819.

## Deliberately excluded

No manuscript images, rendered figures, LaTeX source, bilingual-reader source, reviewer files, superseded result sets, or historical iterations are included. Figure scripts write newly generated files to `reproduced_outputs/`; that directory is not pre-populated in this package.

## Algorithm identifiers

The machine-readable identifiers follow the current manuscript terminology:

- `standard_dqn`
- `full_action_q_dqn`
- `immediate_advantage_dqn`
- `centered_full_action_dqn`
- `double_dqn`
- `double_centered_full_action_dqn`
- `contextual_bandit`

Legacy project or method acronyms are not used in this package.

## Environment

The recorded run used Python 3.13.9, NumPy 2.4.1, pandas 3.0.3, PyTorch 2.8.0 with CUDA 12.9, and Matplotlib 3.10.8 on Windows 11. Install the direct dependencies with:

```powershell
python -m pip install -r dqn_family_satellite_ground/requirements.txt
```

## Verify the package

From the package root:

```powershell
python verify_package.py
python -m unittest discover -s dqn_family_satellite_ground/tests -v
python -m dqn_family_satellite_ground.reviewer_experiments validate --seeds 800-819
```

The first command verifies the full file inventory, SHA-256 checksums, checkpoint counts, embedded model identifiers, and training budgets.

## Reproduce the reported result chain from bundled checkpoints

```powershell
python -m dqn_family_satellite_ground.reviewer_experiments all --seeds 800-819 --resume
python -m dqn_family_satellite_ground.extended_revision --seeds 800-819 --stage all --resume
python -m dqn_family_satellite_ground.transition_audit
python -m dqn_family_satellite_ground.reviewer_reporting
python -m dqn_family_satellite_ground.extended_reporting
python -m dqn_family_satellite_ground.plot_dqn_family_summary
matlab -batch "addpath(fullfile(pwd,'dqn_family_satellite_ground')); plot_reviewer_results_matlab(pwd)"
```

The reporting and plotting commands write derived tables and figures to `reproduced_outputs/` or the documented script output directory. These derived files are not part of the distributed package.

## Retrain in a fresh output directory

To preserve the locked bundled results, direct a from-scratch run to a new directory:

```powershell
python -m dqn_family_satellite_ground.reviewer_experiments all --seeds 800-819 --results reproduced_outputs/full_rerun
python -m dqn_family_satellite_ground.extended_revision --seeds 800-819 --stage all --resume --results reproduced_outputs/full_rerun
```

## Repository and licence status

The permanent public repository URL and release licence have not yet been supplied by the author. They are therefore not invented here, and the manuscript retains an explicit archive-identifier placeholder pending author input.
