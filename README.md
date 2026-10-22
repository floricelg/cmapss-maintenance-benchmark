# C MAPSS failure risk classification benchmark

Experimental code for **Machine Learning for Predictive Maintenance: A Methodologically Consistent Benchmark on NASA C-MAPSS Turbofan Degradation Data**.

The benchmark compares Logistic Regression, Random Forest, HistGradientBoosting, LSTM and same-window RF-LSTM feature fusion on FD001 to FD004. It includes engine-level splitting, training-only normalization, five-fold grouped RF probabilities for fusion, validation-only threshold selection, temporal sensitivity, retrained ablations, grouped permutation importance and held-out-subset transfer.

## Contents

- `scripturi/benchmark.py` fits the models and saves predictions, split membership and training histories.
- `scripturi/summarize_results.py` regenerates the 13 numerical tables, four result figures, utility-weight analysis and paired engine-cluster bootstrap intervals.
- `scripturi/validate_experiments.py` checks labels, engine partitions, grouped RF folds and early-stopping histories.
- `scripturi/download_data.py` downloads or imports the pinned NASA archive and verifies checksums.
- `scripturi/run_pipeline.py` runs a stage using the active Python interpreter.
- `reference_results/` contains the archived numerical summaries and figures for comparison. It is not a cache of trained runs.
- `tests/` contains small protocol checks that require no dataset download or full model training.

Raw NASA data, individual saved predictions, trained checkpoints, manuscript files and reviewer reports are not included. Fresh runs write to `rezultate/`, which is ignored by Git. Word editing and local document-rendering helpers are not needed for these experiments and are excluded.

## Installation

The reference experiments used Python 3.13.7 on CPU, with the versions pinned in `requirements.txt`. Use Python 3.13. A virtual environment is recommended.

Windows PowerShell, from this repository folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

In the commands below, `python` means that virtual environment's interpreter. On Windows, it can be replaced with `.\.venv\Scripts\python.exe`. No PowerShell activation policy change is required.

## Obtain and verify the data

```bash
python scripturi/download_data.py
```

Source: [NASA PCoE Turbofan Engine Degradation Simulation](https://zenodo.org/records/15346912), DOI `10.5281/zenodo.15346912`, file `CMAPSSData.zip`.

If the service refuses or times out the download, download that archive in a browser and import it locally:

```bash
python scripturi/download_data.py --archive "path/to/CMAPSSData.zip"
python scripturi/download_data.py --verify-only
```

Archive SHA256 is `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`. The twelve numeric files are checked against `data_manifest.json`. Network access is needed only to obtain the archive, not for training. The runner no longer requires a second local copy of the NASA dataset.

## Checks and execution

Small protocol checks, without downloading data:

```bash
python -m unittest discover -s tests -v
```

Data audit, without model training:

```bash
python scripturi/run_pipeline.py --stage audit
```

Complete experiment, tables and validation:

```bash
python scripturi/run_pipeline.py --stage all
```

The complete experiment fits all model configurations. It is not an instantaneous check. Run it on a machine with sufficient memory for multivariate window arrays and training. There is no hardware-independent runtime estimate.

Individual stages are `main`, `sensitivity`, `transfer` and `long-window`. The main stage includes branch and observed-cycle ablations. The sensitivity stage already includes the long-window runs. Completed runs are skipped when their `done.json` exists. Do not reuse a results directory after changing the code, input data or configuration. Preserve the old directory under a different name and start a fresh `rezultate/` directory.

The summary and final lineage validation require all 33 configuration-and-seed runs. To regenerate them after completing individual stages:

```bash
python scripturi/summarize_results.py
python scripturi/validate_experiments.py
```

## Experimental protocol

The main task uses 30-cycle windows, stride five and a positive class defined by final-window RUL at most 30 cycles. All three operational settings and 21 sensor channels are retained. Static inputs contain final value, mean, population standard deviation and within-window slope for each channel, plus the observed start and end cycles. Final trajectory length, RUL, engine identity and subset code are not predictors.

The 709 official training engines are split into 567 training and 142 validation engines per seed. The official test files contain 707 engines, of which 690 produce complete 30-cycle windows. FD004 contains 249 training engines and 248 test engines in the numerical files. The source README reverses these counts. The official numerical partitions are not exchanged.

Seeds are 13, 37 and 73. Neural training uses at most 30 epochs, patience five and restoration of the lowest weighted validation-loss checkpoint. Probability thresholds maximize validation utility over 0.05 to 0.95 in steps of 0.01, breaking ties with the lowest threshold. Main utility is `0.7 * sensitivity + 0.3 * specificity`. AUPRC means non-interpolated average precision, not trapezoidal interpolation.

Horizon sensitivity uses 10, 20, 30, 40 and 50 cycles at window length 30. Length sensitivity uses 15, 30 and 60 cycles at horizon 30. These are separate one-factor analyses. Transfer excludes the entire target subset from development. The four utility-weight settings reuse fitted probabilities but select a new validation threshold. The 1,000 paired bootstrap replicates resample entire engines within subsets.

## Reference findings and limitations

The reference pooled utility means are Logistic Regression 0.965, Random Forest 0.936, Gradient Boosting 0.945, LSTM 0.937 and RF-LSTM 0.950. Pooled AUPRC means are 0.836, 0.772, 0.827, 0.751 and 0.792 in the same order. Full values and sample standard deviations are in `reference_results/tabele/table_2.csv`.

The ranking is conditional on representation, warning horizon, window eligibility and the selected decision rule. Static models receive observed cycle indices, while the standalone LSTM receives the 24 measured channels. The cycle ablations assess that difference for RF and Logistic Regression. No exact duplicated main windows were found. Approximate similarities were not removed.

The study does not establish generalization to physical equipment, measured maintenance savings or superiority in inference latency or memory. Transfer experiments remain within the C-MAPSS simulator. CPU libraries, operating systems and numerical implementations can introduce differences in a fresh run. This package does not claim bitwise-identical training on every platform.

## Relation to the manuscript version

The reference summaries correspond to the corrected experimental revision dated 16 October 2026. Packaging on 21 October 2026 changes input-file verification from comparison with a second local dataset copy to pinned SHA256 checks. It also moves Table 13 generation into the numerical summary script, so that all 13 tables can be produced without a Word document. Model fitting, features, seeds, threshold selection and the scientific protocol are retained.

## Publication and citation

The repository owner and coauthors should agree on visibility and a software license before public release. No software license or repository DOI has been assigned in this package. The data source retains its own attribution and terms. Cite the NASA dataset and the article once its final publication details are available. No article DOI or public repository URL is invented here.
