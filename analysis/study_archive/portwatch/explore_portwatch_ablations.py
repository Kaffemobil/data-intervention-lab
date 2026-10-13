from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path('/mnt/data'); OUT=ROOT/'portwatch_ais_study_outputs'
feat=pd.read_csv(OUT/'portwatch_product_month_features.csv',dtype={'hs':str}); feat['origin_period']=pd.PeriodIndex(feat.origin_period,freq='M')
av=pd.read_csv(OUT/'feature_availability_by_month.csv'); av['origin_period']=pd.PeriodIndex(av.origin_period,freq='M')
good=set(av.loc[av.product_complete_fraction>=.8,'origin_period'])
f=pd.read_csv(ROOT/'logistics_pilot_outputs/rolling_forecasts.csv',parse_dates=['origin','target_end']); f=f[(f.model=='logistics')&(f.horizon==3)].copy(); f['hs']=f.series_id.str.extract(r'(\d+)$')[0].str.zfill(2); f['origin_period']=f.origin.dt.to_period('M');f['target_period']=f.target_end.dt.to_period('M');f['residual_m1']=f.actual-f.prediction
BASE=['origin_export_container_pressure','origin_export_cargo_pressure','origin_container_call_pressure','destination_import_container_pressure','destination_import_cargo_pressure','destination_container_call_pressure','route_container_pressure','route_cargo_pressure']
G={
 'full':sum(([x]+[f'{x}_lag{i}' for i in [1,2,3]] for x in BASE),[])+['matched_route_exposure_coverage'],
 'origin_only':sum(([x]+[f'{x}_lag{i}' for i in [1,2,3]] for x in BASE[:3]),[]),
 'destination_only':sum(([x]+[f'{x}_lag{i}' for i in [1,2,3]] for x in BASE[3:6]),[]),
 'route_only':sum(([x]+[f'{x}_lag{i}' for i in [1,2,3]] for x in BASE[6:]),[])+['matched_route_exposure_coverage'],
 'current_only':BASE+['matched_route_exposure_coverage'],
}
AL=[.01,.1,1,10,100,1000]
res=[]
for name,preds in G.items():
 d=f.merge(feat,on=['hs','origin_period'],how='inner'); d=d[d.origin_period.isin(good)&d[preds].notna().all(axis=1)].copy().sort_values(['origin_period','hs'])
 outs=[]
 for cur in sorted(d.origin_period.unique()):
  current=d[d.origin_period==cur]; train=d[d.target_period<=cur]
  ors=sorted(train.origin_period.unique())
  if len(ors)<12: continue
  vo=ors[-6:]; tr=train[~train.origin_period.isin(vo)]; va=train[train.origin_period.isin(vo)]
  if len(tr)<50 or len(va)<20: continue
  sc=StandardScaler().fit(tr[preds]); Xtr=sc.transform(tr[preds]); Xv=sc.transform(va[preds]); ytr=tr.residual_m1.to_numpy(); yv=va.residual_m1.to_numpy()
  ms=[]
  for a in AL:
   m=Ridge(alpha=a).fit(Xtr,ytr); ms.append(np.mean((yv-m.predict(Xv))**2))
  a=AL[int(np.argmin(ms))]
  sc2=StandardScaler().fit(train[preds]); m=Ridge(alpha=a).fit(sc2.transform(train[preds]),train.residual_m1.to_numpy()); corr=m.predict(sc2.transform(current[preds]))
  for rr,cc in zip(current.itertuples(index=False),corr): outs.append((str(cur),rr.actual,rr.prediction,rr.prediction+cc))
 o=pd.DataFrame(outs,columns=['origin','actual','m1','pw']); e1=o.actual-o.m1; ep=o.actual-o.pw; r1=np.sqrt(np.mean(e1**2)); rp=np.sqrt(np.mean(ep**2));
 post=o[pd.PeriodIndex(o.origin,freq='M')>pd.Period('2022-12','M')]; pe1=post.actual-post.m1; pep=post.actual-post.pw; pr1=np.sqrt(np.mean(pe1**2)); prp=np.sqrt(np.mean(pep**2));
 res.append({'model':name,'n':len(o),'rrmse_full':rp/r1,'reduction_full_pct':100*(1-rp/r1),'rrmse_post_covid':prp/pr1,'reduction_post_pct':100*(1-prp/pr1)})
print(pd.DataFrame(res).sort_values('rrmse_full').to_string(index=False))
pd.DataFrame(res).to_csv(OUT/'exploratory_ablation_metrics_h3.csv',index=False)
