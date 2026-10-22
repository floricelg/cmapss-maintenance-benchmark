from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'rezultate'
TABLES=OUT/'tabele'
FIG=OUT/'figuri'
MODELS=['Logistic Regression','Random Forest','Gradient Boosting','LSTM','RF-LSTM']
METRICS=['accuracy','precision','recall','f1','specificity','pme','auprc']
LABELS=['Accuracy','Precision','Recall','F1','Specificity','PME utility','AUPRC']
SEEDS=[13,37,73]

def metrics(y,p,t,alpha=.7):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel()
    recall=tp/(tp+fn);specificity=tn/(tn+fp)
    return dict(accuracy=(tn+tp)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,recall=recall,f1=2*tp/(2*tp+fp+fn),specificity=specificity,pme=alpha*recall+(1-alpha)*specificity,auprc=average_precision_score(y,p),tn=tn,fp=fp,fn=fn,tp=tp)

def threshold(y,p,alpha):
    grid=np.round(np.arange(.05,.951,.01),2)
    score=alpha*(p[y==1,None]>=grid).mean(0)+(1-alpha)*(p[y==0,None]<grid).mean(0)
    return grid[np.argmax(score)]

def summary(df,keys):
    return df.groupby(keys,sort=False)[METRICS].agg(['mean','std']).reset_index()

def formatted(df,keys):
    data=summary(df,keys)
    result=pd.DataFrame({key:data[key] for key in keys})
    for metric,label in zip(METRICS,LABELS):
        result[label]=[f'{m:.3f} ({s:.3f})' for m,s in zip(data[(metric,'mean')],data[(metric,'std')])]
    return result

def main():
    TABLES.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('experiment_runner',Path(__file__).with_name('benchmark.py'))
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    full_parameters={
        'Logistic Regression':runner.LogisticRegression(**runner.CONFIG['logistic'],random_state=13).get_params(),
        'Random Forest':runner.model_rf(13).get_params(),
        'Gradient Boosting':runner.HistGradientBoostingClassifier(**runner.CONFIG['gradient_boosting'],random_state=13).get_params(),
        'seed_note':'random_state equals the current experiment seed. RF fold seeds are seed + fold + 1.',
        'neural':runner.CONFIG['neural'],
        'AdamW_defaults':{'betas':[.9,.999],'eps':1e-8,'amsgrad':False},
    }
    (OUT/'complete_model_parameters.json').write_text(json.dumps(full_parameters,indent=2),encoding='utf-8')
    paths=list((OUT/'runs').glob('*/seed_*/metrics.csv'))
    done=list((OUT/'runs').glob('*/seed_*/done.json'))
    assert len(done)==33, f'Expected 33 completed runs, found {len(done)}'
    frames=[pd.read_csv(path) for path in paths]
    all_metrics=pd.concat(frames,ignore_index=True)
    all_metrics.to_csv(OUT/'all_metrics.csv',index=False)
    main_df=all_metrics[all_metrics.experiment.eq('L30_H30')]
    global_df=main_df[(main_df.subset=='ALL')&main_df.model.isin(MODELS)]
    formatted(global_df,['model']).to_csv(TABLES/'table_2.csv',index=False)
    for number,subset in enumerate(['FD001','FD002','FD003','FD004'],3):
        part=main_df[main_df.subset.eq(subset)&main_df.model.isin(MODELS)]
        formatted(part,['model']).to_csv(TABLES/f'table_{number}.csv',index=False)
    # Audit table includes all raw and window-eligible engine counts.
    counts=pd.read_csv(OUT/'engine_counts.csv');rows=[]
    for subset in ['FD001','FD002','FD003','FD004']:
        a=counts[(counts.subset==subset)&(counts.partition=='train')].iloc[0]
        b=counts[(counts.subset==subset)&(counts.partition=='test')].iloc[0]
        rows.append([subset,a.raw_engines,b.raw_engines,a.window_engines,b.window_engines,b.short_engines,a.windows,b.windows,b.positives])
    pd.DataFrame(rows,columns=['Subset','Raw train engines','Raw test engines','Window train engines','Window test engines','Short test engines','Train windows','Test windows','Test positives']).to_csv(TABLES/'table_1.csv',index=False)
    splits=pd.read_csv(OUT/'split_counts.csv')
    class_weights=[]
    for seed in SEEDS:
        training=splits[(splits.seed==seed)&(splits.partition=='train')]
        n=int(training.windows.sum());pos=int(training.positives.sum());neg=n-pos
        class_weights.append(dict(seed=seed,windows=n,positive=pos,negative=neg,bce_positive_weight=neg/pos,logistic_negative_weight=n/(2*neg),logistic_positive_weight=n/(2*pos)))
    pd.DataFrame(class_weights).to_csv(OUT/'class_weights.csv',index=False)
    split_table=splits.pivot(index=['seed','subset'],columns='partition',values='engines').reset_index()[['seed','subset','train','validation']]
    split_table['Test engines']=split_table.subset.map(counts[counts.partition=='test'].set_index('subset').window_engines)
    split_table.to_csv(TABLES/'table_7.csv',index=False)
    horizon=all_metrics[all_metrics.experiment.isin([f'L30_H{h}' for h in [10,20,30,40,50]])&all_metrics.model.isin(MODELS)&all_metrics.subset.eq('ALL')].copy()
    horizon['RUL horizon']=horizon.experiment.str.extract(r'_H(\d+)').astype(int)
    horizon=horizon.sort_values(['RUL horizon'],kind='stable')
    formatted(horizon,['RUL horizon','model']).to_csv(TABLES/'table_8.csv',index=False)
    window=all_metrics[all_metrics.experiment.isin([f'L{l}_H30' for l in [15,30,60]])&all_metrics.model.isin(MODELS)&all_metrics.subset.eq('ALL')].copy()
    window['Window length']=window.experiment.str.extract(r'L(\d+)').astype(int)
    window=window.sort_values('Window length',kind='stable')
    formatted(window,['Window length','model']).to_csv(TABLES/'table_9.csv',index=False)
    formatted(main_df[main_df.subset.eq('ALL')],['model']).to_csv(TABLES/'table_10.csv',index=False)
    weights=[]
    validation_checks=[]
    metadata=pd.read_csv(OUT/'test_windows_L30.csv')
    for path in paths:
        rows=pd.read_csv(path)
        folder=path.parent
        for row in rows[rows.subset=='ALL'].itertuples():
            saved=np.load(folder/(row.model.replace(' ','_')+'.npz'))
            y=saved['test_y'];p=saved['test_prob'];yv=saved['validation_y'];pv=saved['validation_prob']
            th=threshold(yv,pv,.7)
            assert abs(row.threshold-th)<1e-12
            recomputed=metrics(y,p,th)
            for key,value in recomputed.items():
                assert np.isclose(value,getattr(row,key),rtol=1e-7,atol=1e-8), (folder,row.model,key)
            validation_checks.append(dict(experiment=row.experiment,seed=row.seed,model=row.model,valid=True))
            if row.experiment=='L30_H30' and row.model in MODELS:
                for alpha in [.3,.5,.7,.9]:
                    wt=threshold(yv,pv,alpha)
                    weights.append(dict(seed=row.seed,model=row.model,alpha=alpha,threshold=wt,**metrics(y,p,wt,alpha)))
    weights=pd.DataFrame(weights);weights.to_csv(OUT/'utility_weights_by_seed.csv',index=False)
    wt=weights.groupby(['model','alpha'],sort=False).pme.agg(['mean','std']).reset_index()
    wt['value']=[f'{m:.3f} ({s:.3f})' for m,s in zip(wt['mean'],wt['std'])]
    wt.pivot(index='model',columns='alpha',values='value').reindex(MODELS).reset_index().to_csv(TABLES/'table_11.csv',index=False)
    transfer=all_metrics[all_metrics.experiment.str.contains('_without_')&all_metrics.subset.eq('ALL')].copy()
    transfer['Held-out subset']=transfer.experiment.str.extract(r'without_(FD\d+)')
    formatted(transfer,['Held-out subset','model']).to_csv(TABLES/'table_12.csv',index=False)
    pd.DataFrame(validation_checks).to_csv(OUT/'independent_metric_validation.csv',index=False)
    # All global and subset metrics use the same probability and threshold per run.
    for path in paths:
        rows=pd.read_csv(path)
        if '_without_' in rows.experiment.iloc[0]: continue
        length=int(rows.experiment.iloc[0].split('_')[0][1:])
        meta=pd.read_csv(OUT/f'test_windows_L{length}.csv')
        for row in rows[rows.subset!='ALL'].itertuples():
            saved=np.load(path.parent/(row.model.replace(' ','_')+'.npz'))
            take=meta.iloc[saved['test_index']].subset.to_numpy()==row.subset
            computed=metrics(saved['test_y'][take],saved['test_prob'][take],row.threshold)
            for key,value in computed.items():assert np.isclose(value,getattr(row,key),rtol=1e-7,atol=1e-8)
    # Figures use the corrected numerical evaluation only.
    means=global_df.groupby('model')[METRICS].mean().reindex(MODELS)
    fig,ax=plt.subplots(figsize=(10,5))
    im=ax.imshow(means,vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks(range(7),LABELS,rotation=30,ha='right');ax.set_yticks(range(5),MODELS)
    for i in range(5):
        for j in range(7):ax.text(j,i,f'{means.iloc[i,j]:.3f}',ha='center',va='center',color='black' if means.iloc[i,j]>.65 else 'white',fontsize=9)
    fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout();fig.savefig(FIG/'metrics.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(9.5,6.2))
    for ax,model in zip(axes.ravel(),MODELS):
        row=global_df[global_df.model==model][['tn','fp','fn','tp']].mean()
        matrix=row.to_numpy().reshape(2,2)
        ax.imshow(matrix,cmap='Blues');ax.set_title(model,fontsize=11)
        ax.set_xticks([0,1],['Pred. 0','Pred. 1'],fontsize=9);ax.set_yticks([0,1],['True 0','True 1'],fontsize=9)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.55 else 'black',fontsize=10)
    axes.ravel()[-1].axis('off');fig.tight_layout();fig.savefig(FIG/'confusion.png',dpi=300);plt.close(fig)
    corr=pd.read_csv(OUT/'target_correlations.csv').pivot(index='variable',columns='subset',values='pearson')
    channels=[f'op_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
    corr=corr.reindex(channels)
    fig,ax=plt.subplots(figsize=(6.4,8.5));im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set_xticks(range(4),corr.columns);ax.set_yticks(range(24),channels,fontsize=9)
    for i in range(24):
        for j in range(4):
            v=corr.iloc[i,j];ax.text(j,i,'N/A' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.6 else 'black')
    fig.colorbar(im,ax=ax,shrink=.7);fig.tight_layout();fig.savefig(FIG/'correlations.png',dpi=300);plt.close(fig)
    imp=pd.concat([pd.read_csv(OUT/'runs'/'L30_H30'/f'seed_{seed}'/'rf_validation_permutation.csv').assign(seed=seed) for seed in SEEDS])
    imp.to_csv(OUT/'permutation_importance.csv',index=False)
    rank=imp.groupby('variable').auprc_decrease.agg(['mean','std']).sort_values('mean',ascending=False).head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(rank.index,rank['mean'],xerr=rank['std'],color='#295b85',capsize=3)
    ax.set_xlabel('Decrease in validation AUPRC after grouped permutation');fig.tight_layout();fig.savefig(FIG/'importance.png',dpi=300);plt.close(fig)
    # Engine-cluster paired bootstrap, stratified by FD subset, conditional on the fitted models.
    groups=metadata.group.to_numpy();unique=np.unique(groups);engine_subset=np.array([metadata.loc[metadata.group.eq(g),'subset'].iloc[0] for g in unique])
    counts={}
    for model in MODELS:
        seed_counts=[]
        for seed in SEEDS:
            f=OUT/'runs'/'L30_H30'/f'seed_{seed}'
            saved=np.load(f/(model.replace(' ','_')+'.npz'))
            y=saved['test_y'];p=saved['test_prob'];th=threshold(saved['validation_y'],saved['validation_prob'],.7)
            engine_counts=[]
            for g in unique:
                take=groups==g
                engine_counts.append(confusion_matrix(y[take],p[take]>=th,labels=[0,1]).ravel())
            seed_counts.append(engine_counts)
        counts[model]=np.asarray(seed_counts)
    comparisons=[('Logistic Regression','Random Forest'),('Logistic Regression','LSTM'),('Logistic Regression','RF-LSTM'),('Random Forest','LSTM'),('Random Forest','RF-LSTM')]
    rng=np.random.default_rng(20261016);deltas=[]
    def utility(a):
        tn,fp,fn,tp=a.T
        return (.7*tp/(tp+fn)+.3*tn/(tn+fp)).mean()
    for repeat in range(1000):
        indices=np.concatenate([rng.choice(np.flatnonzero(engine_subset==sub),size=(engine_subset==sub).sum(),replace=True) for sub in ['FD001','FD002','FD003','FD004']])
        deltas.append([utility(counts[a][:,indices].sum(1))-utility(counts[b][:,indices].sum(1)) for a,b in comparisons])
    deltas=np.asarray(deltas)
    bootstrap=[]
    for j,(left,right) in enumerate(comparisons):
        bootstrap.append(dict(comparison=f'{left} minus {right}',difference=utility(counts[left].sum(1))-utility(counts[right].sum(1)),lower_95=np.quantile(deltas[:,j],.025),upper_95=np.quantile(deltas[:,j],.975),replicates=1000))
    pd.DataFrame(bootstrap).to_csv(OUT/'paired_engine_bootstrap.csv',index=False)
    logs=[]
    for path in (OUT/'runs').glob('*/seed_*/*_training.json'):
        info=json.loads(path.read_text());logs.append(dict(experiment=path.parents[1].name,seed=int(path.parent.name.split('_')[1]),mode=path.stem.replace('_training',''),**info))
    pd.DataFrame(logs).to_csv(OUT/'neural_training_summary.csv',index=False)
    # Table 13 uses the same selected sensors and formatting as the manuscript.
    descriptive=pd.read_csv(OUT/'descriptive_statistics.csv')
    descriptive_rows=[]
    for channel in ['sensor_2','sensor_4','sensor_7','sensor_11']:
        for subset in ['FD001','FD002','FD003','FD004']:
            row=descriptive[(descriptive.variable==channel)&(descriptive.subset==subset)].iloc[0]
            descriptive_rows.append([subset,channel,int(row['n']),f'{row["mean"]:.3f} ({row["sd"]:.3f})',f'{row["minimum"]:.3f}',f'{row["maximum"]:.3f}'])
    pd.DataFrame(descriptive_rows,columns=['Subset','Sensor','Windows','Mean (SD)','Minimum','Maximum']).to_csv(TABLES/'table_13.csv',index=False)
    for path in TABLES.glob('table_*.csv'):
        df=pd.read_csv(path)
        text='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
        text+='\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
        path.with_suffix('.md').write_text(text+'\n',encoding='utf-8')
    (OUT/'validation_report.txt').write_text('33 completed experiments. All global and per-subset metrics independently recalculated from saved probabilities. Thresholds selected on validation only. All checks passed.\n',encoding='utf-8')
    print('Validation passed. Tables and figures generated.')

if __name__=='__main__':main()
