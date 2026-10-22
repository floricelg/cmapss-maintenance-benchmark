"""Small protocol checks that do not download data or train full models."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark', ROOT / 'scripturi' / 'benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
data_spec = importlib.util.spec_from_file_location('download_data', ROOT / 'scripturi' / 'download_data.py')
download_data = importlib.util.module_from_spec(data_spec)
data_spec.loader.exec_module(download_data)


class ProtocolTests(unittest.TestCase):
    def test_metric_definition(self):
        result = benchmark.metric(np.array([0, 0, 1, 1]), np.array([.1, .8, .9, .2]), .5)
        self.assertEqual([result[key] for key in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])
        self.assertAlmostEqual(result['pme'], .5)

    def test_threshold_tie_uses_lowest_candidate(self):
        result = benchmark.choose_threshold(np.array([0, 1]), np.array([0., 1.]))
        self.assertEqual(result, .05)

    def test_static_features_and_label_endpoint(self):
        raw = {}
        for subset in benchmark.SUBSETS:
            for partition in ['train', 'test']:
                cycles = np.arange(1, 12)
                values = np.repeat(cycles[:, None], 24, axis=1).astype(float)
                raw[(subset, partition)] = [(1, cycles, values, 11 - cycles)]
        sequence, static, metadata = benchmark.windows(raw, length=6)['train']
        self.assertEqual(sequence.shape, (8, 6, 24))
        self.assertEqual(static.shape, (8, 98))
        np.testing.assert_allclose(static[0, :24], 6)
        np.testing.assert_allclose(static[0, 24:48], 3.5)
        np.testing.assert_allclose(static[0, 48:72], np.std(np.arange(1, 7), ddof=0))
        np.testing.assert_allclose(static[0, 72:96], 1)
        np.testing.assert_array_equal(static[0, 96:], [6, 1])
        self.assertEqual(metadata.iloc[0].rul, 5)
        self.assertEqual(metadata.iloc[1].rul, 0)
        self.assertEqual(metadata.group.nunique(), 4)

    def test_no_rf_branch_is_independent_of_auxiliary_input(self):
        benchmark.set_seed(13)
        model = benchmark.Network('no_rf').eval()
        x = torch.zeros(2, 6, 24)
        with torch.no_grad():
            a = model(x, torch.zeros(2, 1))
            b = model(x, torch.ones(2, 1))
        torch.testing.assert_close(a, b)

    def test_rf_only_branch_is_independent_of_sequence(self):
        model = benchmark.Network('rf_only').eval()
        r = torch.full((2, 1), .5)
        with torch.no_grad():
            a = model(torch.zeros(2, 6, 24), r)
            b = model(torch.ones(2, 6, 24), r)
        torch.testing.assert_close(a, b)

    def test_tampered_data_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='cmapss-test-') as temporary:
            folder = Path(temporary)
            path = folder / 'train_FD001.txt'
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                download_data.verify_files(folder, {'train_FD001.txt': '0' * 64})
            self.assertEqual(path.read_bytes(), b'changed')

    def test_rejects_nonmatching_archive(self):
        with tempfile.TemporaryDirectory(prefix='cmapss-test-') as temporary:
            path = Path(temporary) / 'archive.zip'
            path.write_bytes(b'not the official archive')
            with self.assertRaises(ValueError):
                download_data.import_archive(path, Path(temporary) / 'data', {'archive_sha256': '0' * 64, 'files': {}})


if __name__ == '__main__':
    unittest.main()
