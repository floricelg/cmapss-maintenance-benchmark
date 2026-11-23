"""Validate data lineage, split membership and training histories independently."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'rezultate'

def main():
    rows=[]
    for done in sorted((OUT/'runs').glob('*/seed_*/done.json')):
        folder=done.parent
        info=json.loads(done.read_text())
        tag=info['experiment'];length=int(tag.split('_')[0][1:]);horizon=int(tag.split('_')[1][1:])
        train=pd.read_csv(OUT/f'train_windows_L{length}.csv')
        test=pd.read_csv(OUT/f'test_windows_L{length}.csv')
        membership=np.load(folder/'membership.npz')
        tr=membership['train'];va=membership['validation'];te=membership['test']
        assert set(tr).isdisjoint(va)
        assert set(train.iloc[tr].group).isdisjoint(train.iloc[va].group)
        assert len(tr)==info['train_windows'] and len(va)==info['validation_windows'] and len(te)==info['test_windows']
        if '_without_' in tag:
            target=tag.split('_without_')[1]
            assert not train.iloc[tr].subset.eq(target).any()
            assert not train.iloc[va].subset.eq(target).any()
            assert test.iloc[te].subset.eq(target).all()
        oof=np.load(folder/'rf_oof.npz')
        assert np.array_equal(oof['train_index'],tr)
        assert set(oof['fold'])==set(range(5))
        for gid in train.iloc[tr].group.unique():
            assert len(set(oof['fold'][train.iloc[tr].group.to_numpy()==gid]))==1
        assert np.isfinite(oof['probability']).all()
        for path in folder.glob('*.npz'):
            if path.name in ['membership.npz','rf_oof.npz']:continue
            data=np.load(path)
            assert np.array_equal(data['test_y'],(test.iloc[te].rul.to_numpy()<=horizon).astype(int))
            assert np.array_equal(data['validation_y'],(train.iloc[va].rul.to_numpy()<=horizon).astype(int))
            assert np.array_equal(data['test_index'],te)
            assert np.array_equal(data['validation_index'],va)
            assert np.isfinite(data['test_prob']).all() and ((data['test_prob']>=0)&(data['test_prob']<=1)).all()
        for path in folder.glob('*_training.json'):
            log=json.loads(path.read_text())
            history=pd.read_csv(path.with_suffix('.csv'))
            best=int(history.loc[history.validation_loss.idxmin(),'epoch'])
            assert best==log['best_epoch']
            assert len(history)==log['epochs_run']<=30
            assert np.isclose(history.validation_loss.min(),log['best_validation_loss'])
            assert (folder/(path.stem.replace('_training','')+'_state.pt')).exists()
        rows.append(dict(experiment=tag,seed=info['seed'],split_valid=True,labels_valid=True,oof_groups_valid=True,early_stopping_valid=True))
    assert len(rows)==33
    pd.DataFrame(rows).to_csv(OUT/'lineage_validation.csv',index=False)
    print('33 runs passed split, label, OOF and early-stopping checks.')

if __name__=='__main__':main()
