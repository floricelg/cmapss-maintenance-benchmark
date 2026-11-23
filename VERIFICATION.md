# Distribution checks

The distribution was checked on 21 October 2026 in an isolated local copy using Python 3.13 and the pinned experimental libraries.

- All seven small protocol tests passed.
- Importing the pinned NASA ZIP from a local file verified the archive and all twelve numeric file checksums.
- The audit reproduced engine counts, short-trajectory exclusions, exact-duplicate results, development split counts, descriptive statistics and input-to-target correlations.
- The summary script independently recomputed the saved global and subset metrics and regenerated all 13 numerical tables. Their contents matched the reference CSV tables, including displayed decimal precision.
- All 33 retained experiment-and-seed runs passed the split, label, grouped RF fold and early-stopping lineage checks.
- The feature-construction, model, fitting, threshold and experiment-loop functions are syntactically identical to the reference implementation. The data audit now checks a pinned manifest instead of a second local copy of the dataset.

No models were refitted during packaging. Verification used the existing saved predictions and training histories in an isolated directory. Those raw artifacts are not included in the public package. Fresh fitting is available through `python scripturi/run_pipeline.py --stage all` after installation and data import.

The automatic network download was not validated as an end-to-end successful transfer during this packaging check. The manual-download and local-archive import route was validated. This distinction matters because the external archive service can temporarily limit downloads.
