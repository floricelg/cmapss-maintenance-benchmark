"""C-MAPSS failure-risk classification, sensitivity, ablation and transfer experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'date'
OUT = ROOT / 'rezultate'
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [13, 37, 73]
CHANNELS = [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
FEATURES = [f'{stat}_{c}' for stat in ['last','mean','std','slope'] for c in CHANNELS] + ['end_cycle','start_cycle']
GRID = np.round(np.arange(.05, .951, .01), 2)
CONFIG = dict(window=30, stride=5, horizon=30, validation_fraction=.2, seeds=SEEDS,
    input_channels=CHANNELS, static_features=FEATURES, static_std_ddof=0,
    cycle_descriptors='end_cycle and start_cycle in observed cycles, no lifetime normalization',
    logistic=dict(C=1., solver='lbfgs', max_iter=2000, class_weight='balanced'),
    random_forest=dict(n_estimators=140,max_depth=16,min_samples_leaf=4,max_features='sqrt',class_weight='balanced_subsample',bootstrap=True),
    gradient_boosting=dict(max_iter=180, learning_rate=.06,max_leaf_nodes=31,l2_regularization=.05,early_stopping=False),
    neural=dict(hidden=32,layers=1,dropout=.15,head_hidden=24,optimizer='AdamW',lr=.0015,weight_decay=.0001,
        batch_size=512,max_epochs=30,patience=5,min_delta=0.,clip_norm=1.,pos_weight='n_negative / n_positive on training only'),
    oof_folds=5, thresholds=GRID.tolist(), utility_weights=[.7,.3],standardization='training only, static and flattened sequential observations separately',
    implementation_status='Prospective predictors exclude future-dependent cycle_fraction and subset_code. Uses observed end_cycle and start_cycle only.',
    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=torch.__version__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_raw():
    data = {}
    for subset in SUBSETS:
        for partition in ['train','test']:
            a = np.loadtxt(DATA / f'{partition}_{subset}.txt', dtype=np.float64)
            assert a.shape[1] == 26 and np.isfinite(a).all()
            ids = np.unique(a[:,0]).astype(int)
            assert np.array_equal(ids, np.arange(1,len(ids)+1))
            offsets = np.zeros(len(ids)) if partition == 'train' else np.loadtxt(DATA / f'RUL_{subset}.txt')
            assert len(offsets) == len(ids)
            trajectories = []
            for eid in ids:
                b = a[a[:,0] == eid]
                assert np.array_equal(b[:,1],np.arange(1,len(b)+1))
                rul = b[-1,1] - b[:,1] + offsets[eid-1]
                trajectories.append((eid,b[:,1],b[:,2:],rul))
            data[(subset,partition)] = trajectories
    return data

def windows(raw, length=30):
    result = {}
    for partition in ['train','test']:
        seq,static,meta = [],[],[]
        t = np.arange(length,dtype=float)
        centered=t-t.mean()
        for si,subset in enumerate(SUBSETS):
            for eid,cycles,values,rul in raw[(subset,partition)]:
                for start in range(0,len(values)-length+1,5):
                    end=start+length
                    w=values[start:end]
                    descriptors=np.concatenate([w[-1],w.mean(0),w.std(0,ddof=0),centered @ w / (centered @ centered),[cycles[end-1],cycles[start]]])
                    seq.append(w.astype(np.float32))
                    static.append(descriptors.astype(np.float32))
                    meta.append((subset,eid,1000*(si+1)+eid,int(cycles[start]),int(cycles[end-1]),float(rul[end-1])))
        result[partition] = (np.stack(seq),np.stack(static),pd.DataFrame(meta,columns=['subset','engine','group','start','end','rul']))
    return result

def fingerprint(seq):
    return [hashlib.sha256(np.asarray(w,dtype='<f8').tobytes()).hexdigest() for w in seq]

def audit(raw, data):
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'data_manifest.json').read_text(encoding='utf-8'))
    expected=manifest['files']
    checks=[]
    for name in sorted(expected):
        path=DATA/name
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        same=sha==expected[name]
        checks.append(dict(file=path.name,sha256=sha,identical_to_nasa_zenodo=same))
    pd.DataFrame(checks).to_csv(OUT/'file_integrity.csv',index=False)
    if len(checks)!=12 or not all(row['identical_to_nasa_zenodo'] for row in checks):
        raise ValueError('Input data do not match data_manifest.json. Check the original NASA files.')
    counts=[]
    exclusions=[]
    for subset in SUBSETS:
        for part in ['train','test']:
            trajectories=raw[(subset,part)]
            meta=data[part][2]
            sub=meta[meta['subset']==subset]
            short=[eid for eid,cycles,_,_ in trajectories if len(cycles)<30]
            exclusions.extend(dict(subset=subset,partition=part,engine=eid,cycles=len(cycles)) for eid,cycles,_,_ in trajectories if eid in short)
            counts.append(dict(subset=subset,partition=part,raw_engines=len(trajectories),retained_engines=len(trajectories),window_engines=sub.engine.nunique(),short_engines=len(short),windows=len(sub),positives=int((sub.rul<=30).sum())))
    pd.DataFrame(counts).to_csv(OUT/'engine_counts.csv',index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'excluded_short_engines.csv',index=False)
    # Hash unscaled 64-bit raw sequences separately, including sensors-only audit.
    fingerprints={}
    for part in ['train','test']:
        hashes=[]; sensors=[]
        for subset in SUBSETS:
            for eid,cycles,values,rul in raw[(subset,part)]:
                for start in range(0,len(values)-29,5):
                    w=values[start:start+30]
                    hashes.append(hashlib.sha256(w.astype('<f8').tobytes()).hexdigest())
                    sensors.append(hashlib.sha256(w[:,3:].astype('<f8').tobytes()).hexdigest())
        fingerprints[part]=(np.array(hashes),np.array(sensors))
    duplicates=[]; splits=[]; membership=[]
    train_meta=data['train'][2]
    for seed in SEEDS:
        tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(train_meta,groups=train_meta.group))
        assert not set(train_meta.iloc[tr].group)&set(train_meta.iloc[va].group)
        for label,idx in [('train',tr),('validation',va)]:
            for subset in SUBSETS:
                sub=train_meta.iloc[idx];sub=sub[sub.subset==subset]
                splits.append(dict(seed=seed,subset=subset,partition=label,engines=sub.engine.nunique(),windows=len(sub),positives=int((sub.rul<=30).sum())))
            for gid in sorted(train_meta.iloc[idx].group.unique()):
                membership.append(dict(seed=seed,partition=label,group=int(gid)))
        for channel,hi in [('all_24',0),('sensors_21',1)]:
            h=fingerprints['train'][hi]; ht=fingerprints['test'][hi]
            for left,right,a,b in [('train','validation',h[tr],h[va]),('train','test',h[tr],ht),('validation','test',h[va],ht)]:
                duplicates.append(dict(seed=seed,channels=channel,left=left,right=right,shared_unique_hashes=len(set(a)&set(b)),left_within_duplicates=len(a)-len(set(a)),right_within_duplicates=len(b)-len(set(b))))
    pd.DataFrame(duplicates).to_csv(OUT/'duplicate_audit.csv',index=False)
    pd.DataFrame(splits).to_csv(OUT/'split_counts.csv',index=False)
    pd.DataFrame(membership).to_csv(OUT/'engine_membership.csv',index=False)
    # Descriptive statistics and final-cycle correlations use official training windows only.
    rows=[]; corr=[]
    for subset in SUBSETS:
        take=data['train'][2].subset.to_numpy()==subset
        final=data['train'][0][take,-1].astype(float); y=(data['train'][2].rul.to_numpy()[take]<=30).astype(int)
        for j,ch in enumerate(CHANNELS):
            col=final[:,j]
            rows.append(dict(subset=subset,variable=ch,n=len(col),mean=col.mean(),sd=col.std(ddof=1),minimum=col.min(),q25=np.quantile(col,.25),median=np.median(col),q75=np.quantile(col,.75),maximum=col.max()))
            corr.append(dict(subset=subset,variable=ch,pearson=np.corrcoef(col,y)[0,1] if col.std()>1e-8 else np.nan))
    pd.DataFrame(rows).to_csv(OUT/'descriptive_statistics.csv',index=False)
    pd.DataFrame(corr).to_csv(OUT/'target_correlations.csv',index=False)
    print('AUDIT',counts,flush=True)

def model_rf(seed):
    return RandomForestClassifier(**CONFIG['random_forest'],random_state=seed,n_jobs=4)

def metric(y,p,t,alpha=.7):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel()
    recall=tp/(tp+fn) if tp+fn else 0.
    spec=tn/(tn+fp) if tn+fp else 0.
    return dict(threshold=float(t),accuracy=float((tn+tp)/len(y)),precision=float(tp/(tp+fp)) if tp+fp else 0.,recall=float(recall),f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,specificity=float(spec),pme=float(alpha*recall+(1-alpha)*spec),auprc=float(average_precision_score(y,p)) if y.sum() else np.nan,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n=len(y),positives=int(y.sum()))

def choose_threshold(y,p,alpha=.7):
    # Deterministic tie break: first, lowest threshold on the predefined grid.
    positive=y==1; negative=~positive
    recall=(p[positive,None]>=GRID).mean(0)
    specificity=(p[negative,None]<GRID).mean(0)
    return GRID[np.argmax(alpha*recall+(1-alpha)*specificity)]

class Network(nn.Module):
    def __init__(self, mode='lstm'):
        super().__init__()
        self.mode=mode
        self.lstm=nn.LSTM(24,32,batch_first=True) if mode!='rf_only' else None
        dim=1 if mode=='rf_only' else 33 if mode in ['fusion','no_rf'] else 32
        self.pre_dropout=nn.Dropout(.15)
        self.head=nn.Sequential(nn.Linear(dim,24),nn.ReLU(),nn.Dropout(.15),nn.Linear(24,1))
    def forward(self,x,r):
        if self.mode=='rf_only':
            h=r
        else:
            _,(hidden,_)=self.lstm(x)
            h=self.pre_dropout(hidden[-1])
            if self.mode in ['fusion','no_rf']:
                h=torch.cat([h,r if self.mode=='fusion' else torch.zeros_like(r)],dim=1)
        return self.head(h).squeeze(1)

def neural_predict(model,x,r):
    model.eval(); values=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            values.append(torch.sigmoid(model(torch.from_numpy(x[start:start+1024]),torch.from_numpy(r[start:start+1024,None]))).numpy())
    return np.concatenate(values)

def train_neural(mode,seed,xt,xv,yt,yv,rt,rv,folder):
    set_seed(seed)
    model=Network(mode)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((len(yt)-yt.sum())/yt.sum())))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(rt[:,None]),torch.from_numpy(yt.astype(np.float32))),batch_size=512,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best_loss=np.inf;best=None;wait=0;history=[]
    start_time=time.perf_counter()
    for epoch in range(1,31):
        model.train();total=0
        for x,r,y in loader:
            optimizer.zero_grad(); loss=criterion(model(x,r),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total+=loss.item()*len(y)
        model.eval();val_total=0
        with torch.no_grad():
            for start in range(0,len(xv),1024):
                end=start+1024
                logits=model(torch.from_numpy(xv[start:end]),torch.from_numpy(rv[start:end,None]))
                val_total+=criterion(logits,torch.from_numpy(yv[start:end].astype(np.float32))).item()*len(logits)
        val_loss=val_total/len(xv)
        history.append(dict(epoch=epoch,train_loss=total/len(xt),validation_loss=val_loss))
        if val_loss<best_loss:
            best_loss=val_loss;best=copy.deepcopy(model.state_dict());wait=0;best_epoch=epoch
        else:
            wait+=1
        if wait>=5:
            break
    model.load_state_dict(best)
    pd.DataFrame(history).to_csv(folder/f'{mode}_training.csv',index=False)
    torch.save(best,folder/f'{mode}_state.pt')
    (folder/f'{mode}_training.json').write_text(json.dumps(dict(best_epoch=best_epoch,epochs_run=epoch,best_validation_loss=best_loss,seconds=time.perf_counter()-start_time,parameters=sum(p.numel() for p in model.parameters())),indent=2))
    print('NEURAL',folder.name,mode,'epochs',epoch,'best',best_epoch,'seconds',round(time.perf_counter()-start_time,1),flush=True)
    return model

def run(data,seed,horizon=30,length=30,heldout=None,ablation=False):
    tag=f'L{length}_H{horizon}'+(f'_without_{heldout}' if heldout else '')
    folder=OUT/'runs'/tag/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'done.json').exists():
        print('SKIP',tag,seed,flush=True); return
    started=time.perf_counter()
    set_seed(seed)
    seq,static,meta=data['train']; test_seq,test_static,test_meta=data['test']
    eligible=np.flatnonzero(meta.subset.to_numpy()!=heldout) if heldout else np.arange(len(meta))
    ti,vi=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(eligible,groups=meta.iloc[eligible].group))
    ti,vi=eligible[ti],eligible[vi]
    assert not set(meta.iloc[ti].group)&set(meta.iloc[vi].group)
    test_idx=np.flatnonzero(test_meta.subset.to_numpy()==heldout) if heldout else np.arange(len(test_meta))
    scaler=StandardScaler().fit(static[ti]); xt,xv,xe=[scaler.transform(a).astype(np.float32) for a in [static[ti],static[vi],test_static[test_idx]]]
    seq_scaler=StandardScaler().fit(seq[ti].reshape(-1,24))
    st,sv,se=[seq_scaler.transform(a.reshape(-1,24)).reshape(a.shape).astype(np.float32) for a in [seq[ti],seq[vi],test_seq[test_idx]]]
    yt=(meta.rul.to_numpy()[ti]<=horizon).astype(int);yv=(meta.rul.to_numpy()[vi]<=horizon).astype(int);ye=(test_meta.rul.to_numpy()[test_idx]<=horizon).astype(int)
    np.savez_compressed(folder/'membership.npz',train=ti,validation=vi,test=test_idx,static_mean=scaler.mean_,static_scale=scaler.scale_,sequence_mean=seq_scaler.mean_,sequence_scale=seq_scaler.scale_)
    records=[]
    def save_result(name,pv,pe):
        th=choose_threshold(yv,pv)
        for subset in ['ALL']+([heldout] if heldout else SUBSETS):
            take=np.ones(len(ye),dtype=bool) if subset=='ALL' else test_meta.iloc[test_idx].subset.to_numpy()==subset
            records.append(dict(experiment=tag,seed=seed,model=name,subset=subset,**metric(ye[take],pe[take],th)))
        np.savez_compressed(folder/(name.replace(' ','_')+'.npz'),validation_prob=pv,test_prob=pe,validation_y=yv,test_y=ye,validation_index=vi,test_index=test_idx)
    lr=LogisticRegression(**CONFIG['logistic'],random_state=seed).fit(xt,yt)
    save_result('Logistic Regression',lr.predict_proba(xv)[:,1],lr.predict_proba(xe)[:,1])
    rf=model_rf(seed).fit(xt,yt)
    rv=rf.predict_proba(xv)[:,1].astype(np.float32);re=rf.predict_proba(xe)[:,1].astype(np.float32)
    save_result('Random Forest',rv,re)
    pd.DataFrame(dict(feature=FEATURES,importance=rf.feature_importances_)).to_csv(folder/'rf_importance.csv',index=False)
    gb=HistGradientBoostingClassifier(**CONFIG['gradient_boosting'],random_state=seed).fit(xt,yt)
    save_result('Gradient Boosting',gb.predict_proba(xv)[:,1],gb.predict_proba(xe)[:,1])
    oof=np.full(len(ti),np.nan,dtype=np.float32)
    fold_ids=np.empty(len(ti),dtype=int)
    # All scaling remains inside the outer training partition. RF uses only within-training folds.
    for fold,(fit,hold) in enumerate(GroupKFold(n_splits=5).split(xt,yt,meta.iloc[ti].group)):
        assert not set(meta.iloc[ti[fit]].group)&set(meta.iloc[ti[hold]].group)
        fold_rf=model_rf(seed+fold+1).fit(xt[fit],yt[fit])
        oof[hold]=fold_rf.predict_proba(xt[hold])[:,1]
        fold_ids[hold]=fold
    assert np.isfinite(oof).all()
    np.savez_compressed(folder/'rf_oof.npz',probability=oof,train_index=ti,fold=fold_ids)
    for name,mode in [('LSTM','lstm'),('RF-LSTM','fusion')]:
        neural=train_neural(mode,seed,st,sv,yt,yv,oof,rv,folder)
        save_result(name,neural_predict(neural,sv,rv),neural_predict(neural,se,re))
    if ablation:
        for name,mode in [('Fusion without RF','no_rf'),('Fusion without LSTM','rf_only')]:
            neural=train_neural(mode,seed,st,sv,yt,yv,oof,rv,folder)
            save_result(name,neural_predict(neural,sv,rv),neural_predict(neural,se,re))
        rf_no_age=model_rf(seed).fit(xt[:,:96],yt)
        save_result('RF without cycle descriptors',rf_no_age.predict_proba(xv[:,:96])[:,1],rf_no_age.predict_proba(xe[:,:96])[:,1])
        lr_no_age=LogisticRegression(**CONFIG['logistic'],random_state=seed).fit(xt[:,:96],yt)
        save_result('Logistic Regression without cycle descriptors',lr_no_age.predict_proba(xv[:,:96])[:,1],lr_no_age.predict_proba(xe[:,:96])[:,1])
        # Group-permutation importance on validation, never used for model selection.
        baseline=average_precision_score(yv,rv); rng=np.random.default_rng(seed)
        importance=[]
        for channel in range(24):
            cols=[channel+24*k for k in range(4)]
            drops=[]
            for repeat in range(5):
                pert=xv.copy(); perm=rng.permutation(len(xv));pert[:,cols]=xv[perm][:,cols]
                drops.append(baseline-average_precision_score(yv,rf.predict_proba(pert)[:,1]))
            importance.append(dict(variable=CHANNELS[channel],auprc_decrease=np.mean(drops),within_seed_sd=np.std(drops,ddof=1)))
        cols=[96,97];drops=[]
        for repeat in range(5):
            pert=xv.copy();perm=rng.permutation(len(xv));pert[:,cols]=xv[perm][:,cols]
            drops.append(baseline-average_precision_score(yv,rf.predict_proba(pert)[:,1]))
        importance.append(dict(variable='cycle_descriptors',auprc_decrease=np.mean(drops),within_seed_sd=np.std(drops,ddof=1)))
        pd.DataFrame(importance).to_csv(folder/'rf_validation_permutation.csv',index=False)
    pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False)
    (folder/'done.json').write_text(json.dumps(dict(experiment=tag,seed=seed,train_windows=len(ti),validation_windows=len(vi),test_windows=len(test_idx),seconds=time.perf_counter()-started,ablation=ablation),indent=2))
    print('DONE',tag,seed,'seconds',round(time.perf_counter()-started,1),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','main','sensitivity','transfer','long-window','all'],default='all');args=p.parse_args()
    torch.set_num_threads(4)
    raw=load_raw(); base=windows(raw,30)
    OUT.mkdir(exist_ok=True)
    (OUT/'configuration.json').write_text(json.dumps(CONFIG,indent=2),encoding='utf-8')
    audit(raw,base)
    if args.stage=='audit': return
    if args.stage in ['main','all']:
        for seed in SEEDS:run(base,seed,ablation=True)
    if args.stage in ['sensitivity','all']:
        for horizon in [10,20,40,50]:
            for seed in SEEDS: run(base,seed,horizon=horizon)
        for length in [15,60]:
            current=windows(raw,length)
            current['train'][2].to_csv(OUT/f'train_windows_L{length}.csv',index=False)
            current['test'][2].to_csv(OUT/f'test_windows_L{length}.csv',index=False)
            for seed in SEEDS:run(current,seed,length=length)
    if args.stage in ['transfer','all']:
        for heldout in SUBSETS:
            for seed in SEEDS:run(base,seed,heldout=heldout)
    if args.stage=='long-window':
        current=windows(raw,60)
        current['train'][2].to_csv(OUT/'train_windows_L60.csv',index=False)
        current['test'][2].to_csv(OUT/'test_windows_L60.csv',index=False)
        for seed in SEEDS:run(current,seed,length=60)
    base['train'][2].to_csv(OUT/'train_windows_L30.csv',index=False)
    base['test'][2].to_csv(OUT/'test_windows_L30.csv',index=False)

if __name__=='__main__':main()
