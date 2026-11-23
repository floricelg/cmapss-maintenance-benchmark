# NASA C-MAPSS Failure-Risk Classification Benchmark

Reproducibility materials for **“Machine Learning for Predictive Maintenance: A Methodologically Consistent Benchmark on NASA C-MAPSS Turbofan Degradation Data.”**

This project compares Logistic Regression (LR), Random Forest (RF), histogram-based Gradient Boosting (GB), LSTM, and RF–LSTM on the NASA C-MAPSS FD001–FD004 subsets. It studies **binary near-term failure-risk classification**, not continuous remaining-useful-life (RUL) regression. Its contribution is a controlled, leakage-aware comparison and an investigation of how experimental and maintenance-decision assumptions affect model rankings; it does not propose a fundamentally new learning algorithm.

**Repository:** https://github.com/floricelg/cmapss-maintenance-benchmark

## Main features

- Engine-level training/validation separation and untouched official test trajectories.
- Training-only preprocessing; grouped, out-of-fold RF probabilities for the fusion branch.
- Validation-only classification-threshold selection and multiple experimental seeds.
- Warning-horizon and observation-window sensitivity studies.
- RF–LSTM branch and observed-cycle feature ablations; grouped permutation importance.
- Cross-subset transfer within the simulated C-MAPSS environment.
- Paired engine-cluster bootstrap comparisons.
- **Early-fusion versus late-fusion RF–LSTM** comparison under matched reference conditions.
- **Decision robustness:** precision–recall curves and AUPRC, fixed-threshold sensitivity, false-positive/false-negative trade-offs, and hypothetical cost-ratio analysis.

## Project structure

| Path | Purpose |
| --- | --- |
| `scripturi/benchmark.py` | Main benchmark, model training, predictions, and experimental runs. |
| `scripturi/summarize_results.py` | Numerical summaries, tables, figures, and statistical comparisons. |
| `scripturi/validate_experiments.py` | Protocol and experiment validation. |
| `scripturi/download_data.py` | Obtain/import the NASA data archive and verify integrity. |
| `scripturi/run_pipeline.py` | Execute the supported benchmark stages. |
| `scripturi/run_early_fusion.py` | Additional early-fusion RF–LSTM experiment, where included in the working version. |
| `scripturi/analyze_decision_robustness.py` | Post-hoc decision and hypothetical-cost analyses from saved predictions. |
| `reference_results/` | Archived summary results and reference figures, where distributed. |
| `rezultate/` | Local run-level outputs, predictions, checkpoints, and generated analyses. |
| `tests/` | Lightweight protocol checks, where distributed. |
| `REVIEWER_REPRODUCIBILITY.md` | Detailed inventory of what the **published repository** actually distributes; update alongside this README. |

> **Artifact availability:** A local experimental archive and the public GitHub repository are not necessarily identical. In the working archive used for the revision, `rezultate/runs/` contains per-model `.npz` predictions, neural checkpoints (`.pt`), and training histories for several configurations. Before claiming these files are *publicly downloadable*, verify that the corresponding files are present in the published GitHub repository or a linked release. Do not describe locally available artifacts as publicly archived unless they have been uploaded.

## Software setup

The original reference environment was documented as Python **3.13.7** on CPU, with dependencies in `requirements.txt`. The revision was also executed using a Conda environment named `cmapss-final`. For maximum reproducibility, use the versions pinned in the repository's `requirements.txt`; exact results can still vary with operating system, library versions, and hardware.

### Windows PowerShell — virtual environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The commands below assume `python` refers to the configured project environment. If using Conda, activate the environment first:

```powershell
conda activate cmapss-final
```

## Data acquisition and verification

```bash
python scripturi/download_data.py
```

Source: [NASA PCoE Turbofan Engine Degradation Simulation, Zenodo record 15346912](https://zenodo.org/records/15346912), DOI: `10.5281/zenodo.15346912`; archive: `CMAPSSData.zip`.

If automatic download fails:

```bash
python scripturi/download_data.py --archive "path/to/CMAPSSData.zip"
python scripturi/download_data.py --verify-only
```

Expected archive SHA256:

```text
74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f
```

The numeric source files are checked against `data_manifest.json`. NASA data attribution and licensing remain separate from any license applied to this code.

## Running the benchmark

From the repository root:

```bash
python -m unittest discover -s tests -v
python scripturi/run_pipeline.py --stage audit
python scripturi/run_pipeline.py --stage all
```

The full pipeline retrains multiple configurations and may require substantial CPU time and memory. Individual supported stages include `main`, `sensitivity`, `transfer`, and `long-window`. Completed runs may be skipped when a `done.json` marker is present. **Do not reuse a completed results directory after changing code, data, or configuration**; preserve it separately and begin a new run directory.

After completing the relevant runs:

```bash
python scripturi/summarize_results.py
python scripturi/validate_experiments.py
```

The original reference workflow documents **33 configuration-and-seed runs** for its full summary and lineage checks. Additional reviewer experiments should be documented and verified separately; do not assume they are included in that original count.

## Experimental protocol

The reference configuration uses window length **L = 30**, stride **5**, and binary positive labels for final-window **RUL ≤ 30 cycles** (warning horizon **H = 30**). All three operational-setting channels and 21 sensor channels are retained.

Static features summarize each measured channel by final value, mean, population standard deviation, and within-window slope; observed start and end cycles are also included. Engine identity, subset identifier, final trajectory length, and RUL are not used as predictors.

The official training data contain **709 engines**; each experimental seed uses **567 training** and **142 validation** engines. The official test data contain **707 engines**, of which **690** yield complete 30-cycle windows under the reference protocol. Seeds: **13, 37, 73**.

The primary utility is:

```text
PME utility = 0.7 × sensitivity + 0.3 × specificity
```

The `0.7` sensitivity weight is an **illustrative decision preference**, not an empirically calibrated maintenance-cost ratio. Classification thresholds are selected on validation data over `0.05`–`0.95` in increments of `0.01`, with ties resolved toward the lower threshold. AUPRC is reported as non-interpolated average precision.

The main sensitivity experiments consider warning horizons **10, 20, 30, 40, 50** cycles and observation-window lengths **15, 30, 60** cycles. The utility-weight study examines multiple sensitivity–specificity preferences. Transfer experiments leave the target C-MAPSS subset out of model development; this is **not** validation on independent industrial data.

## RF–LSTM fusion comparison

The original **late-fusion** architecture processes 24 sequential channels through an LSTM with 32 hidden units, then concatenates the resulting representation with a scalar RF risk probability before classification (33-dimensional head input).

The additional **early-fusion** architecture repeats the RF probability across the 30 time steps and appends it as a 25th sequential input channel; the LSTM processes the combined sequence before classification.

Both configurations use the same reference setting (**L = 30, H = 30**), engine-level splits, seeds (**13, 37, 73**), and corresponding RF training/validation/test probabilities. The neural models are trained separately, and thresholds are selected using validation predictions only.

| Metric (mean across three seeds) | Late fusion | Early fusion |
| --- | ---: | ---: |
| Recall | 0.932 | 0.942 |
| F1 | 0.594 | 0.518 |
| PME utility | 0.941 | 0.943 |
| AUPRC | 0.789 | 0.771 |

Early fusion has slightly higher mean recall and utility in this comparison, while late fusion has higher mean F1 and AUPRC. Neither approach is established as universally superior.

To rerun the additional experiment **if the script is present in your checkout**, use:

```bash
python scripturi/run_early_fusion.py
```

Check the script's local arguments and expected input paths before execution. It depends on the reference benchmark outputs and should not overwrite the original run results.

## Decision robustness and hypothetical cost analysis

After the **L30/H30** run predictions have been generated, run from the repository root:

```bash
python scripturi/analyze_decision_robustness.py
```

This is a **post-processing analysis**: it loads saved validation and test predictions and does not retrain the models. Outputs are written to:

```text
rezultate/reviewer4_robustness/
```

Generated files include:

| File | Description |
| --- | --- |
| `figure_R4_precision_recall.png` | Representative held-out precision–recall curves (seed 13). |
| `table_R4_auprc_by_seed.csv` | Per-model AUPRC for each of the three seeds. |
| `table_R4_auprc_summary.csv` | Mean and standard deviation of AUPRC across seeds. |
| `figure_R4_fp_fn_tradeoff.png` | False alarms versus missed positives across fixed thresholds. |
| `table_R4_fixed_thresholds.csv` | Sensitivity, specificity, FP and FN at each fixed threshold. |
| `figure_R4_decision_regions.png` | Descriptive best-model regions by threshold and sensitivity weight. |
| `table_R4_decision_regions.csv` | Underlying decision-region results. |
| `figure_R4_hypothetical_costs.png` | Mean hypothetical classification cost across cost ratios. |
| `table_R4_hypothetical_costs.csv` | Per-seed, per-model cost results and validation-selected thresholds. |
| `INTERPRETATION_NOTES.txt` | Interpretation and limitations of these outputs. |

### Hypothetical cost definition

For each model, seed, and cost ratio, the script selects the threshold minimizing **validation** cost and applies that threshold unchanged to the official held-out test predictions:

```text
normalized cost per 1,000 windows = 1,000 × (FP + r × FN) / N
```

where `r = C_FN / C_FP` is one of **1, 2, 5, 10, 20, 50**, `C_FP` is normalized to one unit, and `N` is the number of evaluated test windows. Lower values indicate lower *hypothetical classification penalty*.

The investigated scenarios favor Logistic Regression in mean normalized cost, but the fixed-threshold decision-region analysis also shows that **model rankings change with threshold and sensitivity–specificity preferences**. These are complementary findings, not evidence of universal superiority.

**Important limitations:** These are **illustrative, uncalibrated costs**, not euros, dollars, measured downtime, or verified maintenance savings. FP and FN count **prediction windows**, not distinct engine failures or maintenance work orders. Overlapping windows may represent repeated warnings for the same engine. The decision-region plot uses held-out test outcomes **descriptively only**, not to select a model or threshold for deployment. PR curves display seed 13; the AUPRC tables cover all three seeds.

## Interpretation and limitations

- Logistic Regression's favorable results in the reference setting are conditional on the data representation, warning horizon, thresholding protocol, and decision objective.
- Greater architectural complexity does not automatically imply higher utility or better threshold-free discrimination.
- C-MAPSS contains simulated degradation trajectories. It does not represent industrial maintenance interventions, equipment-specific cost structures, measurement uncertainty, or operational concept drift in full.
- Binary near-failure classification is not equivalent to continuous RUL prediction.
- Cross-subset transfer remains within the same simulation family and is not real-world external validation.
- Performance and runtime can vary across platforms; bitwise-identical retraining is not guaranteed.

## Reproducibility and public-release checklist

Before publication or updating the manuscript's **Data Availability Statement**, verify that the **public GitHub repository** (not merely a local ZIP or ignored `rezultate/` directory) includes or links to:

1. Source scripts, environment requirements, data manifest, and reproduction instructions.
2. The reference numerical tables, figures, configuration, and split summaries.
3. The early-fusion experiment script and its results.
4. The decision-robustness script and the figures/tables generated for the revision.
5. Any original-run predictions, neural checkpoints, and per-epoch histories **explicitly claimed as publicly available**; otherwise, describe them as regenerable or available separately, not as archived.
6. An updated `REVIEWER_REPRODUCIBILITY.md` that agrees with this README and the manuscript.

A local archive examined during the revision contained run-level `.npz` prediction files, `.pt` neural checkpoints, and training-history files. **Their presence in a local archive does not verify that they are committed to GitHub.** If Git excludes `rezultate/`, upload appropriate artifacts through a documented release or another archival mechanism before stating that they are publicly available.

## Citation

When using the data, cite the NASA C-MAPSS source and its corresponding archive record. When using this benchmark, cite the associated *Applied Sciences* manuscript once its final bibliographic details and DOI are available. Do not infer a manuscript DOI or software license from this README.

**Project:** [cmapss-maintenance-benchmark](https://github.com/floricelg/cmapss-maintenance-benchmark)
