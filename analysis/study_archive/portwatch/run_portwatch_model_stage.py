from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path('/mnt/data'); OUT=ROOT/'portwatch_ais_study_outputs'
FORECASTS=ROOT/'logistics_pilot_outputs/rolling_forecasts.csv'
FEAT=OUT/'portwatch_product_month_features.csv'
AVAIL=OUT/'feature_availability_by_month.csv'
BASE=[
'origin_export_container_pressure','origin_export_cargo_pressure','origin_container_call_pressure',
'destination_import_container_pressure','destination_import_cargo_pressure','destination_container_call_pressure',
'route_container_pressure','route_cargo_pressure']
predictors=[]
for f in BASE: predictors += [f]+[f'{f}_lag{i}' for i in [1,2,3]]
predictors+=['matched_route_exposure_coverage']
ALPHAS=[0.01,0.1,1,10,100,1000]
COVID_START=pd.Period('2020-03','M'); COVID_END=pd.Period('2022-12','M')

feat=pd.read_csv(FEAT,dtype={'hs':str})
feat['origin_period']=pd.PeriodIndex(feat.origin_period,freq='M')
av=pd.read_csv(AVAIL)
av['origin_period']=pd.PeriodIndex(av.origin_period,freq='M')
good_months=set(av.loc[av.product_complete_fraction>=0.8,'origin_period'])
feat=feat[feat.origin_period.isin(good_months) & feat[predictors].notna().all(axis=1)].copy()

f=pd.read_csv(FORECASTS,parse_dates=['origin','target_end'])
f=f[f.model=='logistics'].copy()
f['hs']=f.series_id.str.extract(r'(\d+)$')[0].str.zfill(2)
f['origin_period']=f.origin.dt.to_period('M'); f['target_period']=f.target_end.dt.to_period('M')
f['residual_m1']=f.actual-f.prediction
j=f.merge(feat,on=['hs','origin_period'],how='inner')
print('joined rows',len(j),'months',j.origin_period.min(),j.origin_period.max(),'products',j.hs.nunique())

rows=[]; alphas=[]
for h in [1,3,6]:
    d=j[j.horizon==h].copy().sort_values(['origin_period','hs'])
    for cur in sorted(d.origin_period.unique()):
        current=d[d.origin_period==cur].copy()
        train=d[d.target_period<=cur].copy()
        tr_orig=sorted(train.origin_period.unique())
        if len(tr_orig)<12 or current.empty: continue
        val_orig=tr_orig[-6:]
        itr=train[~train.origin_period.isin(val_orig)]; iva=train[train.origin_period.isin(val_orig)]
        if len(itr)<50 or len(iva)<20: continue
        Xtr=itr[predictors].to_numpy(float); ytr=itr.residual_m1.to_numpy(float)
        Xv=iva[predictors].to_numpy(float); yv=iva.residual_m1.to_numpy(float)
        sc=StandardScaler().fit(Xtr); Xtrs=sc.transform(Xtr); Xvs=sc.transform(Xv)
        mses={}
        for a in ALPHAS:
            m=Ridge(alpha=a).fit(Xtrs,ytr); pr=m.predict(Xvs); mses[a]=float(np.mean((yv-pr)**2))
        best=min(ALPHAS,key=lambda a: mses[a])
        Xa=train[predictors].to_numpy(float); ya=train.residual_m1.to_numpy(float); Xc=current[predictors].to_numpy(float)
        sc2=StandardScaler().fit(Xa); m=Ridge(alpha=best).fit(sc2.transform(Xa),ya); corr=m.predict(sc2.transform(Xc))
        bias=float(ya.mean())
        for k,r in enumerate(current.itertuples(index=False)):
            rows.append({'horizon':h,'origin_period':str(cur),'target_period':str(r.target_period),'hs':r.hs,
                         'series_id':r.series_id,'series_name':r.series_name,'actual':r.actual,
                         'm1_prediction':r.prediction,'m1_error':r.actual-r.prediction,
                         'pw_correction':float(corr[k]),'pw_prediction':float(r.prediction+corr[k]),
                         'bias_correction':bias,'bias_prediction':float(r.prediction+bias),
                         'alpha':best,'n_train':len(train),'n_train_origins':len(tr_orig)})
        ar={'horizon':h,'origin_period':str(cur),'alpha':best,'n_train':len(train),'n_train_origins':len(tr_orig)}
        ar.update({f'cv_mse_{a}':mses[a] for a in ALPHAS}); alphas.append(ar)

pred=pd.DataFrame(rows); pred.to_csv(OUT/'rolling_portwatch_forecasts.csv',index=False)
pd.DataFrame(alphas).to_csv(OUT/'rolling_alpha_selection.csv',index=False)
pred['op']=pd.PeriodIndex(pred.origin_period,freq='M')

def metrics_for(sub,label):
    out=[]
    for h,g in sub.groupby('horizon'):
        e1=(g.actual-g.m1_prediction).to_numpy(); ep=(g.actual-g.pw_prediction).to_numpy(); eb=(g.actual-g.bias_prediction).to_numpy()
        r1=np.sqrt(np.mean(e1*e1)); rp=np.sqrt(np.mean(ep*ep)); rb=np.sqrt(np.mean(eb*eb))
        out.append({'sample':label,'horizon':int(h),'n':len(g),'n_months':g.origin_period.nunique(),
                    'rmse_m1':r1,'rmse_portwatch':rp,'rrmse_portwatch':rp/r1,'rmse_reduction_pct':100*(1-rp/r1),
                    'rmse_bias_only':rb,'rrmse_bias_only':rb/r1,'pw_vs_bias_rrmse':rp/rb})
    return out
M=[]
M+=metrics_for(pred,'full')
M+=metrics_for(pred[(pred.op<COVID_START)|(pred.op>COVID_END)],'ex_covid')
M+=metrics_for(pred[pred.op<COVID_START],'pre_covid')
M+=metrics_for(pred[(pred.op>=COVID_START)&(pred.op<=COVID_END)],'covid')
M+=metrics_for(pred[pred.op>COVID_END],'post_covid')
metrics=pd.DataFrame(M); metrics.to_csv(OUT/'aggregate_metrics.csv',index=False)
cat=[]
for (h,hs),g in pred.groupby(['horizon','hs']):
    e1=(g.actual-g.m1_prediction).to_numpy(); ep=(g.actual-g.pw_prediction).to_numpy(); r1=np.sqrt(np.mean(e1*e1)); rp=np.sqrt(np.mean(ep*ep))
    cat.append({'horizon':h,'hs':hs,'series_name':g.series_name.iloc[0],'n':len(g),'rmse_m1':r1,'rmse_portwatch':rp,'rrmse':rp/r1,'rmse_reduction_pct':100*(1-rp/r1)})
pd.DataFrame(cat).to_csv(OUT/'category_metrics.csv',index=False)
print('\nMETRICS\n',metrics.to_string(index=False))
print('\nALPHA counts\n',pd.DataFrame(alphas).groupby(['horizon','alpha']).size().to_string())
print('\nPRIMARY categories\n',pd.DataFrame(cat).query('horizon==3').sort_values('rrmse').to_string(index=False))
