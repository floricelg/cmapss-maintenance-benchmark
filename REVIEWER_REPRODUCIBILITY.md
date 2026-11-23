# Reproducibility Materials for Peer Review

This document describes the reproducibility materials associated with the revised manuscript, *Machine Learning for Predictive Maintenance: A Methodologically Consistent Benchmark on NASA C-MAPSS Turbofan Degradation Data*.

It distinguishes archived experimental summaries, executable analysis scripts, and original run-level artifacts. The purpose is to support independent verification while clearly identifying the limitations of the distributed materials.

## 1. Experimental Scope

The benchmark evaluates Logistic Regression, Random Forest, HistGradientBoosting, LSTM, and RF–LSTM feature fusion using the NASA C-MAPSS FD001–FD004 datasets.

The experimental framework includes:

- Engine-level training and validation partitions and official held-out test trajectories.
- Training-only normalization and grouped out-of-fold Random Forest predictions.
- Validation-only classification threshold selection.
- Warning-horizon and observation-window sensitivity analyses.
- Model-component and observed-cycle ablations.
- Grouped permutation importance and cross-subset transfer evaluation.
- Paired engine-level bootstrap confidence intervals.
- Early-fusion versus late-fusion RF–LSTM comparison.
- Precision–recall and AUPRC analysis.
- Classification-threshold and sensitivity-weight decision analysis.
- False-positive versus false-negative trade-off analysis.
- Hypothetical maintenance-related classification cost analysis.

The reference configuration uses a window length of 30 cycles, a warning horizon of 30 cycles, and experimental seeds 13, 37, and 73.

## 2. Archived Reproducibility Materials

The repository documentation identifies the following reference materials:

- **Source code:** `scripturi/`, covering model training, preprocessing, evaluation, reporting, and validation.
- **Dataset provenance:** `data_manifest.json` and `scripturi/download_data.py`.
- **Software environment:** `requirements.txt`.
- **Model configurations:** `reference_results/complete_model_parameters.json` and `configuration.json`.
- **Engine-level partitions:** `reference_results/engine_membership.csv` and `split_counts.csv`.
- **Classification thresholds and evaluation metrics:** `reference_results/all_metrics.csv`.
- **Neural training summaries:** `reference_results/neural_training_summary.csv`.
- **Numerical summaries and figures:** `reference_results/tabele/`, `reference_results/figuri/`, and associated reference result files.
- **Protocol verification:** `tests/` and the available experimental validation scripts.

These materials document the benchmark protocol and provide reference outputs for comparison with independently generated results.

## 3. Additional Experiments Introduced During Revision

### 3.1 Early-Fusion and Late-Fusion RF–LSTM Comparison

An additional early-fusion RF–LSTM configuration was evaluated against the original late-fusion architecture.

Both configurations used the same reference setting (L = 30, H = 30), engine-level partitions, Random Forest prediction inputs, and experimental seeds.

The comparison investigated whether introducing Random Forest probabilities into the sequential input, rather than combining them with the learned LSTM representation, affected predictive performance.

The results demonstrated differences across evaluation metrics without establishing consistent superiority of either fusion configuration.

### 3.2 Precision–Recall and AUPRC Evaluation

Precision–recall curves were generated to complement the maintenance-oriented weighted utility analysis.

The illustrated precision–recall curves use seed 13, while the quantitative AUPRC evaluation covers all three experimental seeds.

AUPRC is evaluated using non-interpolated average precision.

### 3.3 Classification Threshold and Decision Sensitivity

An additional descriptive analysis investigated model rankings across fixed classification thresholds from 0.05 to 0.95 and sensitivity weights from 0.1 to 0.9.

The decision-region analysis uses held-out predictions to describe how model rankings change across operating conditions.

**The held-out decision-region results were not used to select classification thresholds or optimize models.**

### 3.4 False-Positive and False-Negative Trade-offs

The relationship between false alarms and missed positive predictions was examined across classification thresholds.

Results were normalized per 1,000 held-out prediction windows and averaged across the three experimental seeds.

These counts represent window-level classification errors, not unique engine failures or maintenance interventions.

### 3.5 Hypothetical Cost-Sensitive Evaluation

Six hypothetical missed-positive-to-false-alarm cost ratios were investigated: 1:1, 2:1, 5:1, 10:1, 20:1, and 50:1.

For each model, experimental seed, and cost ratio, classification thresholds were selected exclusively using validation predictions and subsequently evaluated on the held-out official test data.

Normalized classification costs were calculated per 1,000 prediction windows.

The analysis examines robustness under alternative assumed error costs. The ratios are illustrative and are not derived from measured industrial maintenance expenses.

## 4. Reproducing the Benchmark

Follow the installation and dataset verification instructions in `README.md`.

Run the protocol checks:

```bash
python -m unittest discover -s tests -v
```

Run the complete benchmark:

```bash
python scripturi/run_pipeline.py --stage all
```

Regenerate the main numerical summaries and validate the experimental lineage:

```bash
python scripturi/summarize_results.py
python scripturi/validate_experiments.py
```

The additional decision-robustness analysis can be executed using:

```bash
python scripturi/analyze_decision_robustness.py
```

This additional script requires the corresponding saved prediction artifacts from the reference configuration.

The early-fusion experiment must likewise be reproduced using its associated experimental implementation and the required Random Forest prediction inputs.

The complete benchmark may require substantial CPU time and memory. Results may vary across software environments and numerical implementations.

## 5. Original Run-Level Artifacts

The archived numerical summaries should not be confused with complete original training runs.

The earlier public package did not distribute the complete original per-epoch neural training histories, per-window predicted probabilities, or fitted model checkpoints.

The executable workflow can generate new run-level artifacts in `rezultate/`, but independently regenerated artifacts are not automatically identical to those from the original completed experiments.

Any claim that original prediction files, checkpoints, or training histories are publicly available must be supported by the files actually present in the released repository or an explicitly linked supplementary archive.

## 6. Interpretation and External Validity

The experiments use simulated degradation trajectories from NASA C-MAPSS.

The findings are limited to the evaluated benchmark, data partitions, prediction targets, model configurations, and decision assumptions.

Cross-subset transfer within C-MAPSS does not constitute independent real-world validation.

The hypothetical cost analysis does not demonstrate actual maintenance savings, and window-level prediction errors do not correspond directly to distinct maintenance actions.

The results therefore provide reproducible experimental evidence concerning model behavior and decision sensitivity, rather than proof of operational superiority in industrial maintenance environments.

## 7. Materials for Editorial Verification

If the Editorial Office requests original run-level evidence, a separate archive of retained original experimental outputs may be supplied, subject to availability.

Such an archive should include:

- Saved prediction arrays and associated labels.
- Training and validation membership information.
- Selected classification thresholds.
- Neural training histories and model checkpoints, where retained.
- A manifest identifying filenames, experiment configurations, and file hashes.

The provenance of original and independently regenerated artifacts should remain clearly distinguished.

## 8. Relationship to the Manuscript

The reproducibility materials are intended to support the methodological descriptions, experimental comparisons, and additional analyses reported in the revised manuscript.

The manuscript remains the authoritative source for the final reported numerical results, figure numbering, and scientific interpretation.

Before public release, the repository inventory should be checked against the actual committed files to ensure that every referenced script, table, figure, and supplementary artifact is available or explicitly identified as requiring regeneration.
