import pandas as pd, numpy as np, json, hashlib, time
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA=Path('/mnt/data'); OUT=DATA/'logistics_pilot_outputs'; OUT.mkdir(exist_ok=True)

def load_bls(path):
    d=pd.read_csv(path,sep='\t'); d.columns=d.columns.str.strip(); d['series_id']=d.series_id.astype(str).str.strip(); d['period']=d.period.astype(str).str.strip()
    d=d[d.period.str.fullmatch(r'M(0[1-9]|1[0-2])',na=False)].copy(); d['month']=d.period.str[1:].astype(int)
    d['date']=pd.to_datetime(dict(year=d.year,month=d.month,day=1))+pd.offsets.MonthEnd(0); d['value']=pd.to_numeric(d.value,errors='coerce')
    return d[['series_id','date','value']]

def ridge_predict(Xtr,ytr,Xp,alpha):
    sc=StandardScaler(); Xs=sc.fit_transform(Xtr); xp=sc.transform(Xp)
    m=Ridge(alpha=alpha); m.fit(Xs,ytr); return m.predict(xp), m

h=load_bls(DATA/'ei_harmonized_import.tsv'); b=load_bls(DATA/'ei_bea_import.tsv')
meta=pd.read_csv(DATA/'ei_series.tsv',sep='\t'); meta.columns=meta.columns.str.strip(); meta['series_id']=meta.series_id.astype(str).str.strip(); meta['index_code']=meta.index_code.astype(str).str.strip()
ip=meta[meta.index_code.eq('IP')].copy(); ip['suffix']=ip.series_id.str.replace('EIUIP','',regex=False); ip=ip[ip.suffix.str.fullmatch(r'\d{2}',na=False)].copy(); ip['hs2']=ip.suffix.astype(int)
universe=ip[(ip.begin_year<=2004)&(ip.hs2>=28)][['series_id','suffix','series_name','begin_year','base_period']].sort_values('suffix')
universe.to_csv(OUT/'primary_universe.csv',index=False); sids=universe.series_id.tolist(); labels=universe.set_index('series_id').series_name.to_dict()
hp=h[h.series_id.isin(sids+['EIUIP27'])].pivot(index='date',columns='series_id',values='value').sort_index(); bea=b[b.series_id.eq('EIUIREXFUELS')].pivot(index='date',columns='series_id',values='value').sort_index()
g=pd.read_excel(DATA/'gscpi_conv/gscpi_data.xlsx',sheet_name='GSCPI Monthly Data').iloc[:,0:2]; g.columns=['date','gscpi']; g['date']=pd.to_datetime(g.date,errors='coerce'); g['gscpi']=pd.to_numeric(g.gscpi,errors='coerce'); g=g.dropna().set_index('date').sort_index(); g.index=g.index+pd.offsets.MonthEnd(0)
idx=pd.date_range(max(hp.index.min(),bea.index.min(),g.index.min()),min(hp.index.max(),bea.index.max(),g.index.max()),freq='ME'); c=pd.DataFrame(index=idx); c['exfuel']=bea['EIUIREXFUELS'].reindex(idx); c['fuel']=hp['EIUIP27'].reindex(idx); c['gscpi']=g.gscpi.reindex(idx)
for col in ['exfuel','fuel']:
    lp=np.log(c[col]); c[f'{col}_m1']=100*lp.diff(); c[f'{col}_m3']=100*lp.diff(3); c[f'{col}_m12']=100*lp.diff(12)
for j in range(7): c[f'gscpi_l{j}']=c.gscpi.shift(j)
c['gscpi_mean3']=c.gscpi.rolling(3).mean(); c['gscpi_chg3']=c.gscpi-c.gscpi.shift(3); c['month_sin']=np.sin(2*np.pi*c.index.month/12); c['month_cos']=np.cos(2*np.pi*c.index.month/12)
BASE=['own_m1_l0','own_m1_l1','own_m1_l2','own_m1_l3','own_m1_l6','own_m1_l12','own_m3','own_m12','exfuel_m1','exfuel_m3','exfuel_m12','fuel_m1','fuel_m3','fuel_m12','month_sin','month_cos']
LOG=BASE+[f'gscpi_l{j}' for j in range(7)]+['gscpi_mean3','gscpi_chg3']; ALPHAS=np.array([.01,.1,1,10,100,1000.]); H=[1,3,6]; TEST=pd.Timestamp('2017-01-31')

def make_df(sid,hh):
    d=c.copy(); lp=np.log(hp[sid].reindex(d.index)); m1=100*lp.diff(); d['own_m1_l0']=m1
    for j in [1,2,3,6,12]: d[f'own_m1_l{j}']=m1.shift(j)
    d['own_m3']=100*lp.diff(3); d['own_m12']=100*lp.diff(12); d['target']=100*(lp.shift(-hh)-lp); d['target_end']=d.index+pd.offsets.MonthEnd(hh)
    return d

def tune(d,features):
    dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]'); X=d[features].to_numpy(float); y=d.target.to_numpy(float); valid=np.isfinite(X).all(1)&np.isfinite(y)
    mse=np.zeros(len(ALPHAS)); n=np.zeros(len(ALPHAS))
    for vy in range(2012,2017):
        vs=np.datetime64(f'{vy}-01-31'); ve=np.datetime64(f'{vy}-12-31'); tr=valid&(tend<vs); va=valid&(dates>=vs)&(dates<=ve)&(tend<=np.datetime64('2016-12-31'))
        if tr.sum()<60 or va.sum()<4: continue
        sc=StandardScaler(); Xt=sc.fit_transform(X[tr]); Xv=sc.transform(X[va]); yt=y[tr]; yv=y[va]
        for k,a in enumerate(ALPHAS):
            m=Ridge(alpha=a).fit(Xt,yt); e=yv-m.predict(Xv); mse[k]+=(e@e); n[k]+=len(e)
    score=np.where(n>0,mse/n,np.inf); return float(ALPHAS[np.argmin(score)]),score

records=[]; arecs=[]; start=time.time()
for hh in H:
  for si,sid in enumerate(sids):
    d=make_df(sid,hh); dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]')
    for mn,features in [('baseline',BASE),('logistics',LOG)]:
      alpha,scores=tune(d,features); arecs.append({'series_id':sid,'horizon':hh,'model':mn,'alpha':alpha,**{f'cv_mse_{a}':v for a,v in zip(ALPHAS,scores)}})
      X=d[features].to_numpy(float); y=d.target.to_numpy(float); valid=np.isfinite(X).all(1)&np.isfinite(y); originmask=(dates>=np.datetime64('2017-01-31'))&(tend<=dates.max())&valid
      for oi in np.flatnonzero(originmask):
        tr=valid&(tend<=dates[oi])
        if tr.sum()<100: continue
        pred,_=ridge_predict(X[tr],y[tr],X[[oi]],alpha)
        records.append({'series_id':sid,'origin':pd.Timestamp(dates[oi]),'target_end':pd.Timestamp(tend[oi]),'horizon':hh,'model':mn,'prediction':float(pred[0]),'actual':float(y[oi]),'n_train':int(tr.sum()),'alpha':alpha})
    print('done',hh,sid,round(time.time()-start,1),flush=True)

fc=pd.DataFrame(records); alphas=pd.DataFrame(arecs); fc['series_name']=fc.series_id.map(labels); fc.to_csv(OUT/'rolling_forecasts.csv',index=False); alphas.to_csv(OUT/'ridge_alpha_selection.csv',index=False)

def metrics(df,sample):
    o=[]
    for (hh,m),z in df.groupby(['horizon','model']):
      e=z.actual-z.prediction; o.append({'sample':sample,'horizon':hh,'model':m,'n':len(z),'rmse':np.sqrt(np.mean(e*e)),'mae':np.mean(abs(e)),'bias':np.mean(z.prediction-z.actual),'sign_accuracy':np.mean(np.sign(z.prediction)==np.sign(z.actual))})
    return o
ex=fc[(fc.target_end<pd.Timestamp('2020-03-31'))|(fc.origin>=pd.Timestamp('2023-01-31'))]; pre=fc[fc.target_end<pd.Timestamp('2020-03-31')]; post=fc[fc.origin>=pd.Timestamp('2023-01-31')]
met=pd.DataFrame(metrics(fc,'full')+metrics(ex,'ex_covid')+metrics(pre,'pre_covid')+metrics(post,'post_covid')); met.to_csv(OUT/'aggregate_metrics.csv',index=False)
p=fc.pivot_table(index=['series_id','series_name','origin','target_end','horizon','actual'],columns='model',values='prediction').reset_index().dropna(); p['se_base']=(p.actual-p.baseline)**2; p['se_log']=(p.actual-p.logistics)**2; p['ae_base']=abs(p.actual-p.baseline); p['ae_log']=abs(p.actual-p.logistics); p['se_diff']=p.se_log-p.se_base; p.to_csv(OUT/'paired_forecast_errors.csv',index=False)
cat=[]
for (hh,sid,name),z in p.groupby(['horizon','series_id','series_name']):
 rb=np.sqrt(z.se_base.mean()); rl=np.sqrt(z.se_log.mean()); cat.append({'horizon':hh,'series_id':sid,'series_name':name,'n':len(z),'rmse_baseline':rb,'rmse_logistics':rl,'rrmse':rl/rb,'rmse_reduction_pct':100*(1-rl/rb),'mae_reduction_pct':100*(1-z.ae_log.mean()/z.ae_base.mean())})
cat=pd.DataFrame(cat); cat.to_csv(OUT/'category_metrics.csv',index=False)

def boot_rr(hh,sample,B=3000,block=6,seed=20260824):
 z=p[p.horizon.eq(hh)].copy()
 if sample=='ex_covid': z=z[(z.target_end<pd.Timestamp('2020-03-31'))|(z.origin>=pd.Timestamp('2023-01-31'))]
 elif sample=='pre': z=z[z.target_end<pd.Timestamp('2020-03-31')]
 elif sample=='post': z=z[z.origin>=pd.Timestamp('2023-01-31')]
 ag=z.groupby('origin').agg(sb=('se_base','sum'),sl=('se_log','sum'),n=('se_base','size')).sort_index(); sb=ag.sb.values; sl=ag.sl.values; nn=ag.n.values; N=len(ag); point=np.sqrt(sl.sum()/nn.sum())/np.sqrt(sb.sum()/nn.sum())
 rng=np.random.default_rng(seed+hh+len(sample)); maxs=max(1,N-block+1); vals=np.empty(B)
 for b in range(B):
  ids=[]
  while len(ids)<N:
   s=int(rng.integers(0,maxs)); ids.extend(range(s,min(s+block,N)))
  ids=np.array(ids[:N]); vals[b]=np.sqrt(sl[ids].sum()/nn[ids].sum())/np.sqrt(sb[ids].sum()/nn[ids].sum())
 lo,hi=np.quantile(vals,[.025,.975]); return {'horizon':hh,'sample':sample,'n_months':N,'n_cells':len(z),'rrmse':point,'rmse_reduction_pct':100*(1-point),'rrmse_ci_low':lo,'rrmse_ci_high':hi,'prob_rrmse_lt_1':np.mean(vals<1)}
boot=pd.DataFrame([boot_rr(hh,s) for hh in H for s in ['full','ex_covid','pre','post']]); boot.to_csv(OUT/'bootstrap_rrmse.csv',index=False)
mon=p.groupby(['horizon','origin'],as_index=False).agg(se_base=('se_base','mean'),se_log=('se_log','mean')); mon['rmse_ratio_month']=np.sqrt(mon.se_log/mon.se_base); mon['loss_improvement']=mon.se_base-mon.se_log; mon.to_csv(OUT/'monthly_loss_comparison.csv',index=False)
# Live forecasts
live=[]; latest=pd.Timestamp('2026-07-31')
for hh in H:
 for sid in sids:
  d=make_df(sid,hh); dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]')
  oi=np.where(dates==np.datetime64('2026-07-31'))[0][0]
  for mn,features in [('baseline',BASE),('logistics',LOG)]:
   a=float(alphas[(alphas.series_id.eq(sid))&(alphas.horizon.eq(hh))&(alphas.model.eq(mn))].alpha.iloc[0]); X=d[features].to_numpy(float); y=d.target.to_numpy(float); valid=np.isfinite(X).all(1)&np.isfinite(y); tr=valid&(tend<=dates[oi]);
   if not np.isfinite(X[oi]).all(): continue
   pred,_=ridge_predict(X[tr],y[tr],X[[oi]],a); rec={'forecast_id':f'US-{sid.replace("EIUIP","HS")}-MXP-20260824-H{hh}-{mn}-v1','issued_at':'2026-08-24','information_month':'2026-07-31','series_id':sid,'series_name':labels[sid],'horizon_months':hh,'target_month':(latest+pd.offsets.MonthEnd(hh)).date().isoformat(),'model':mn,'forecast_log_pct':float(pred[0]),'alpha':a,'n_train':int(tr.sum())}
   hist=fc[(fc.series_id.eq(sid))&(fc.horizon.eq(hh))&(fc.model.eq(mn))]; resid=(hist.actual-hist.prediction).dropna().values
   if len(resid)>=30:
    rec.update({'lower80':float(pred[0]+np.quantile(resid,.1)),'upper80':float(pred[0]+np.quantile(resid,.9)),'lower95':float(pred[0]+np.quantile(resid,.025)),'upper95':float(pred[0]+np.quantile(resid,.975)),'calibration_n':len(resid)})
   live.append(rec)
live=pd.DataFrame(live); live.to_csv(OUT/'experimental_live_forecasts_2026-08-24.csv',index=False)
pri=boot[(boot.horizon.eq(3))&(boot['sample'].eq('full'))].iloc[0]; ex3=boot[(boot.horizon.eq(3))&(boot['sample'].eq('ex_covid'))].iloc[0]; success=bool(pri.rrmse<=.95 and pri.rrmse_ci_high<1 and ex3.rrmse<.98)
summary={'primary_horizon_months':3,'primary_rrmse':float(pri.rrmse),'primary_rmse_reduction_pct':float(pri.rmse_reduction_pct),'primary_95ci':[float(pri.rrmse_ci_low),float(pri.rrmse_ci_high)],'ex_covid_rrmse':float(ex3.rrmse),'ex_covid_rmse_reduction_pct':float(ex3.rmse_reduction_pct),'predeclared_success':success,'interpretation':'PASS' if success else 'FAIL','scope_note':'Tests aggregate GSCPI incremental forecasting value, not the unavailable-in-runtime granular Census port/network extension.'}
json.dump(summary,open(OUT/'decision_summary.json','w'),indent=2)
hashes=[]
for fn in ['ei_harmonized_import.tsv','ei_bea_import.tsv','ei_series.tsv','gscpi_data.xlsx','logistics_price_pilot_fast.py']:
 pp=DATA/fn
 if pp.exists(): hashes.append({'file':fn,'sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'bytes':pp.stat().st_size})
pd.DataFrame(hashes).to_csv(OUT/'input_hashes.csv',index=False)
print('\nMETRICS\n',met.to_string(index=False)); print('\nBOOT\n',boot.to_string(index=False)); print('\nSUMMARY\n',json.dumps(summary,indent=2)); print('\nCAT H3\n',cat[cat.horizon.eq(3)].sort_values('rmse_reduction_pct',ascending=False)[['series_id','series_name','rmse_reduction_pct','rrmse']].to_string(index=False))
