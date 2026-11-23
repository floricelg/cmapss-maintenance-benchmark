# Reproducibility materials for peer review

This file inventories what is **already distributed** in this repository and distinguishes it from run-level artifacts that can be regenerated.

## Publicly archived materials

- Source code: `scripturi/` (training, preprocessing, evaluation, reporting, validation).
- Pinned data provenance and integrity: `data_manifest.json`, `scripturi/download_data.py`.
- Software environment: `requirements.txt`.
- Full model settings: `reference_results/complete_model_parameters.json` and `configuration.json`.
- Engine-level membership for reference splits: `reference_results/engine_membership.csv`, with counts in `split_counts.csv`.
- **Selected decision thresholds**: the `threshold` column of `reference_results/all_metrics.csv`, indexed by experiment, seed, model and subset. This file also contains the reported performance metrics.
- Neural training **summaries**: `reference_results/neural_training_summary.csv` (best epoch, epochs run, best validation loss, duration and parameter counts).
- Numerical tables, figures, statistical comparisons and audit reports: `reference_results/tabele/`, `figuri/`, and other CSV/TXT files in `reference_results/`.

## Artifacts not included in the public archive

The complete per-epoch neural training histories, per-window predicted probabilities and fitted model checkpoints from the **original completed runs** are not distributed. `reference_results/` is an archive of summaries, not a cache of trained runs. The original-run verification described in `VERIFICATION.md` used saved predictions and histories that are not part of this ZIP.

The executable workflow can produce new run-level artifacts in `rezultate/`; a new run is not the same thing as providing the exact artifacts from the original runs. Do not describe newly regenerated artifacts as original-run artifacts without validating their provenance.

## Independent reproduction

Follow the commands in `README.md`. After obtaining and verifying the NASA archive, run:

```bash
python -m unittest discover -s tests -v
python scripturi/run_pipeline.py --stage all
```

The complete pipeline retrains all experiment configurations and can require substantial CPU time and memory. Compare new summaries with `reference_results/`; results may vary across platforms.

## Editorial Office package

If the Editorial Office requests the exact original-run artifacts, provide a separate archive of the retained original `rezultate/` outputs, if available. Include a manifest of filenames and hashes, and check for dataset licensing and privacy considerations. Do not imply that the exact original predictions or complete per-epoch histories are already present in this public repository.
