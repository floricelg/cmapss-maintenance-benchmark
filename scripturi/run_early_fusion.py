"""Standalone early-fusion comparison for the existing C-MAPSS L30_H30 runs.

Run from repository root: python scripturi/run_early_fusion.py
Does not modify existing metrics.csv or done.json files.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as b

def main():
    raw = b.load_raw()
    data = b.windows(raw, length=30)
    seq, static, meta = data["train"]
    test_seq, test_static, test_meta = data["test"]
    output = b.OUT / "reviewer4_early_fusion"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in b.SEEDS:
        original = b.OUT / "runs" / "L30_H30" / f"seed_{seed}"
        for name in ("membership.npz", "rf_oof.npz", "Random_Forest.npz"):
            if not (original / name).is_file():
                raise FileNotFoundError(original / name)
        membership = np.load(original / "membership.npz")
        ti, vi, ei = (membership[k] for k in ("train", "validation", "test"))
        assert np.array_equal(ei, np.arange(len(test_meta))), "Test index order differs"
        assert not set(meta.iloc[ti].group) & set(meta.iloc[vi].group)
        static_scaler = StandardScaler().fit(static[ti])
        seq_scaler = StandardScaler().fit(seq[ti].reshape(-1, 24))
        for label, actual, recorded in (
            ("static mean", static_scaler.mean_, membership["static_mean"]),
            ("static scale", static_scaler.scale_, membership["static_scale"]),
            ("sequence mean", seq_scaler.mean_, membership["sequence_mean"]),
            ("sequence scale", seq_scaler.scale_, membership["sequence_scale"]),
        ):
            if not np.allclose(actual, recorded, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Seed {seed}: {label} does not match original run")
        st, sv, se = [
            seq_scaler.transform(arr.reshape(-1, 24)).reshape(arr.shape).astype(np.float32)
            for arr in (seq[ti], seq[vi], test_seq[ei])
        ]
        yt = (meta.rul.to_numpy()[ti] <= 30).astype(int)
        yv = (meta.rul.to_numpy()[vi] <= 30).astype(int)
        ye = (test_meta.rul.to_numpy()[ei] <= 30).astype(int)
        oof = np.load(original / "rf_oof.npz")
        rf = np.load(original / "Random_Forest.npz")
        if not np.array_equal(oof["train_index"], ti):
            raise ValueError(f"Seed {seed}: RF OOF training order mismatch")
        if not np.array_equal(rf["validation_index"], vi) or not np.array_equal(rf["test_index"], ei):
            raise ValueError(f"Seed {seed}: RF prediction index mismatch")
        if not np.array_equal(rf["validation_y"], yv) or not np.array_equal(rf["test_y"], ye):
            raise ValueError(f"Seed {seed}: RF prediction label mismatch")
        rt = oof["probability"].astype(np.float32)
        rv = rf["validation_prob"].astype(np.float32)
        re = rf["test_prob"].astype(np.float32)
        if not all(np.isfinite(v).all() for v in (rt, rv, re)):
            raise ValueError(f"Seed {seed}: non-finite RF predictions")
        if not (len(rt) == len(st) and len(rv) == len(sv) and len(re) == len(se)):
            raise ValueError(f"Seed {seed}: inconsistent array lengths")
        folder = output / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        print(f"TRAIN Early Fusion seed {seed}", flush=True)
        model = b.train_neural("early_fusion", seed, st, sv, yt, yv, rt, rv, folder)
        pv = b.neural_predict(model, sv, rv)
        pe = b.neural_predict(model, se, re)
        threshold = b.choose_threshold(yv, pv)
        np.savez_compressed(folder / "Early_Fusion_RF-LSTM.npz",
                            validation_prob=pv, test_prob=pe,
                            validation_y=yv, test_y=ye,
                            validation_index=vi, test_index=ei)
        for subset in ["ALL"] + b.SUBSETS:
            mask = np.ones(len(ye), dtype=bool) if subset == "ALL" else test_meta.iloc[ei].subset.to_numpy() == subset
            rows.append(dict(experiment="L30_H30", seed=seed, model="Early Fusion RF-LSTM",
                             subset=subset, **b.metric(ye[mask], pe[mask], threshold)))
        pd.DataFrame(rows).to_csv(output / "metrics_early_fusion.csv", index=False)
        print(f"DONE seed={seed} threshold={threshold:.2f}", flush=True)
    print(f"Saved {output / 'metrics_early_fusion.csv'}", flush=True)

if __name__ == "__main__":
    main()
