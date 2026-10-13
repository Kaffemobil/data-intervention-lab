"""Reproduce the 32-code vector-quantized sampling experiment.

Input: data/vq_modeling_table.csv
Split: origin_period < 2025-01 train; >= 2025-01 future holdout.
Codebook: MiniBatchKMeans(k=32) fit on standardized train features only.
Selection: uniform random vs VQ-balanced (one-per-code first, then sqrt-size allocation).
Downstream model: Ridge predicting the frozen baseline's 3-month residual.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data'/'vq_modeling_table.csv')
fixed=['hs','origin_period','m1_error']
features=[c for c in df.columns if c not in fixed]
train=df[df.origin_period<'2025-01'].copy(); test=df[df.origin_period>='2025-01'].copy()
sc=StandardScaler().fit(train[features])
Xtr=sc.transform(train[features]); Xte=sc.transform(test[features])
ytr=train.m1_error.to_numpy(); yte=test.m1_error.to_numpy()

# Hyperparameter is selected on a strictly earlier validation slice, then frozen.
mask_dev=train.origin_period<'2024-01'
mask_val=train.origin_period>='2024-01'
alphas=[0.1,1,10,100,1000,10000]
val=[]
for a in alphas:
    m=Ridge(alpha=a).fit(Xtr[mask_dev],ytr[mask_dev])
    val.append((a,mean_squared_error(ytr[mask_val],m.predict(Xtr[mask_val]))**0.5))
alpha=min(val,key=lambda x:x[1])[0]

K=32
km=MiniBatchKMeans(n_clusters=K,random_state=7,n_init=20,batch_size=256).fit(Xtr)
codes=km.labels_

def vq_sample(codes,n,rng):
    clusters=[c for c in range(K) if np.any(codes==c)]
    chosen=[]
    if n>=len(clusters):
        for c in clusters:
            inds=np.where(codes==c)[0]
            chosen.append(int(rng.choice(inds)))
        remaining=n-len(chosen)
    else:
        cs=rng.choice(clusters,size=n,replace=False)
        for c in cs:
            inds=np.where(codes==c)[0]
            chosen.append(int(rng.choice(inds)))
        return np.asarray(chosen,int)
    if remaining:
        chosen_set=set(chosen)
        available={c:[int(i) for i in np.where(codes==c)[0] if int(i) not in chosen_set] for c in clusters}
        base_weights=np.array([np.sqrt(len(available[c])+1e-9) for c in clusters],float)
        for _ in range(remaining):
            valid_mask=np.array([len(available[c])>0 for c in clusters])
            if not valid_mask.any(): break
            w=base_weights*valid_mask; w=w/w.sum()
            c=int(rng.choice(clusters,p=w)); j=int(rng.integers(len(available[c])))
            chosen.append(available[c].pop(j))
    return np.asarray(chosen,int)

out=[]
for frac in [.25,.5,.75]:
    n=round(frac*len(train))
    for seed in range(200):
        rng=np.random.default_rng(seed)
        idx_random=rng.choice(len(train),n,replace=False)
        idx_vq=vq_sample(codes,n,rng)
        for method,idx in [('Random',idx_random),('VQ-balanced',idx_vq)]:
            model=Ridge(alpha=alpha).fit(Xtr[idx],ytr[idx])
            pred=model.predict(Xte)
            out.append([frac,seed,method,mean_squared_error(yte,pred)**0.5,len(np.unique(codes[idx]))/K])
out=pd.DataFrame(out,columns=['budget','seed','method','test_rmse','code_coverage'])
print('selected alpha:',alpha)
print(out.groupby(['budget','method']).agg({'test_rmse':['mean','std'],'code_coverage':'mean'}))
