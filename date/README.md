# NASA input files

Run `python scripturi/download_data.py` from the repository root. The script downloads the pinned NASA archive and imports only the twelve numeric input files. If downloading is unavailable, obtain `CMAPSSData.zip` from https://zenodo.org/records/15346912 and run `python scripturi/download_data.py --archive "path/to/CMAPSSData.zip"`.

Required files are `train_FD001.txt` through `train_FD004.txt`, `test_FD001.txt` through `test_FD004.txt`, and `RUL_FD001.txt` through `RUL_FD004.txt`. `data_manifest.json` records their SHA256 checksums. Existing files with different checksums are not overwritten.

The data remain attributed to NASA PCoE. This repository does not redistribute the raw archive or assign a separate license to it. The numerical FD004 files have 249 training engines and 248 test engines, despite the reversed counts in the accompanying source README.
