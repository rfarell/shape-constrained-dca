# Shape-constrained Bayesian decline curves

Code, frozen configurations, fitted models, and saved scores for

> R. Farell, J. E. Bickel, and C. Bajaj. *Shape-Constrained Bayesian Decline Curves for Joint
> Oil, Gas, and Water Forecasting From Short Production Histories in the Bakken.* Manuscript
> submitted to Geoenergy Science and Engineering, 2026.

The model learns nonnegative continuous-time oil, gas, and water rates for each well and
integrates them into volumes. It samples complete joint trajectories conditioned on the static
attributes and whatever reports a well has at the forecast date. Every interval, cumulative,
ratio, and tail in the paper comes from those joint samples.

## Contents

| Path | What it holds |
| --- | --- |
| `src/joint_production/` | Model, observation law, fitting, samplers, comparators, scores |
| `scripts/` | Command-line entry points; each has a docstring with usage |
| `configs/` | Experiment settings, frozen evaluation protocols, run queues |
| `models/ensemble/` | The five fitted members of the width-diverse ensemble (seeds 42–46) |
| `models/comparators/` | Positive-basis and conditional DCA networks (seeds 42–46) |
| `results/` | Saved scores and diagnostics behind every reported number |
| `data/` | Data description, raw-file checks, and the accepted split (hashes only) |
| `tests/` | Unit and invariant tests |

Phase order is oil, gas, water throughout. Oil and water are in barrels, gas in thousand cubic
feet, and time in days.

## Install

Python 3.11 or later is required.

```
pip install -e ".[dev]"
make test
```

The tests take about 20 seconds on a CPU. The reported
runs used Python 3.11.14, PyTorch 2.10.0 with CUDA 13.0, NumPy 2.3.5, SciPy 1.17.0, and
scikit-learn 1.8.0 on one NVIDIA GB10 GPU (Linux, aarch64), with BLAS and OpenMP thread counts
set to 1.

## Data

The Bakken export is not included because redistribution rights have not been established.
`data/README.md` gives the file's SHA-256, size, columns, and units. With the file at
`data/raw/bakken.csv`, `make prepare` checks it, curates it, and confirms that the rebuilt split
is byte-identical to `data/split_manifest.json`.

## Reproducing the paper

**From the saved results (no data or GPU needed).**

```
make tables            # outputs/results_bundle.md: every table, each with its source file
make summary-figures   # outputs/figures/: accuracy, reliability, ensemble, terminal-decline figures
```

**With the data.**

```
make prepare           # raw-file check, curation, split check
make data-summary      # outputs/data_summary/: the data tables
make figures           # outputs/figures/: model-based figures from the shipped checkpoints
                       # (validation wells only; add DEVICE=cuda on a GPU)
```

Test forecasts from the shipped ensemble members, one per seed, then scoring:

```
for s in 42 43 44 45 46; do
  python scripts/depletion_dev.py repredict --model models/ensemble/bayes_arch_seed${s}_model.pt \
    --label bayes_arch --device cuda --split test \
    --protocol configs/depletion_arch_protocol.toml --output outputs/test_arch_s${s}
done
python scripts/depletion_dev.py score --mixtures --runs outputs/test_arch_s4{2,3,4,5,6} \
  --output outputs/score_test_arch
```

Compare the member and `bayes_arch_mix5` entries with
`results/scores/arch/score_test_full/scores.json`.

**Refitting.** `python scripts/depletion_queue.py configs/<queue>.toml --outputs outputs` runs
each job in a queue and scores it. Test queues read the validation checkpoints from
`outputs/`. They also require a protocol file whose status is `frozen`.

| Stage | Commands |
| --- | --- |
| Comparators (positive basis, conditional DCA, contextual neural, forest), fitted on training wells, with validation forecasts | queue `comparators` |
| Comparator calibration and test forecasts | `depletion_competitor_forecasts.py`, then `depletion_spread_calibration.py` |
| v1 (prespecified) | queues `final_val`, then `depletion_calibrate.py`, then `final_test` |
| Per-well decline curves | queues `final_val_dca`, `final_test_dca`, `scaling` |
| v2 (persistent report error) | queues `v2_val`, `v2_test` |
| Width-diverse ensemble | `ensemble_sweep.py` (development holdout), queues `arch_val`, `arch_test` |
| Sensitivity, ablations, and report-law screen | queues `taumax`, `ablate`, `report_law` |
| Analyses | `depletion_hindcast.py`, `depletion_criteria.py`, `depletion_properties.py`, `depletion_report_properties.py`, `depletion_quickmix.py`, `ensemble_diversity.py`, `depletion_sbc.py`, `paper_taumax_sensitivity.py`, `paper_posterior_check.py` |

## Where the reported numbers live

| Paper item | Saved result | Rebuilt by |
| --- | --- | --- |
| Main validation and test scores, paired gains, calibration, P-notation reliability | `results/scores/arch/`, `results/scores/{val,test}/` | `make tables` |
| Data tables | `results/data_summary.json` | `make data-summary` |
| Accuracy, reliability, ensemble, and terminal-decline figures | `results/figures/summary_figures.json` | `make summary-figures` |
| Updating, decline, mode, phase-coupling, attribute, and correlation figures; forecast card | `results/figures/figures.json` | `make figures` |
| v1 scores, calibration, ablations, per-well scaling | `results/v1/` | `make tables` |
| v2 calibration and prediction streams | `results/v2/`, `results/scores/streams/`, `results/scores/test_quick/` | `make tables` |
| Report-law screen (supplementary ablation table, lower panel) | `results/v2/report_law_screen/` | read directly; refit with queue `report_law` |
| Terminal-decline sensitivity | `results/scores/taumax/` | `make tables` |
| Ensemble development study | `results/scores/diversity_dev/` | `make tables` |
| Latent and reported shape checks | `results/scores/arch/*properties*`, `results/v1/analysis/` | `make tables` |
| Importance-sampling diagnostics and posterior check | `results/v1/sbc/`, `results/scores/{arch/fits,v2_fits,posterior_check}/` | `make tables` |

## Provenance

- This release is a reduced copy of the code that produced the paper. Options, report laws,
  model variants, and scripts that no reported result uses were removed. The ensemble was fitted
  with the full code. On a deterministic subsample of the curated data, run on a CPU, every
  released script gives forecasts and scores bit-identical to those of the full code, and fit
  records that differ only by omitting the removed options. The shipped ensemble and comparator
  checkpoints give bit-identical forecasts under both. The run-manifest `source_hashes` in
  `results/scores/arch/fits/` therefore describe the fitting code, not these files.
- v1, v2, and the shipped comparator networks were fitted by earlier versions of the model and
  scoring code, which are not released. Their saved scores reproduce every reported number.
  Refitting them with this code (queues `final_val`, `v2_val`, `comparators`) need not reproduce
  those numbers exactly.
- The saved result files were edited in four ways; no value changed:
  - machine-specific paths were rewritten relative to this repository;
  - large files were written without indentation;
  - the `evaluation_split` label, written as `train_internal_holdout` for every holdout at the
    time, was corrected for validation and test scores (`evaluation_split_note` marks each one);
  - the run-manifest `source_hashes` were restricted to files that ship in this release.

  `make tables` gives the manuscript's results bundle; only file paths differ, one
  fixed-length excerpt shows more of its record because the paths are shorter, and a closing
  pointer to unreleased development notes is omitted.
  `results/figures/summary_figures.json` and `results/data_summary.json` were regenerated from
  the released files, and their values equal the originals.
- Frozen protocols and experiment configurations are byte-identical to the files whose SHA-256
  the results record, so their comments still refer to the original run directories and files.

## License

The code is released under the MIT License (`LICENSE`). The license does not cover the Bakken
export.
