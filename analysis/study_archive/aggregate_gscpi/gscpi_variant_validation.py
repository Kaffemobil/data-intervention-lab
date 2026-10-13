import pandas as pd, numpy as np, json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
DATA=Path('/mnt/data'); OUT=DATA/'logistics_pilot_outputs'

def load_bls(p):
 d=pd.read_csv(p,sep='\t'); d.columns=d.columns.str.strip(); d['series_id']=d.series_id.astype(str).str.strip(); d['period']=d.period.astype(str).str.strip(); d=d[d.period.str.fullmatch(r'M(0[1-9]|1[0-2])',na=False)].copy(); d['month']=d.period.str[1:].astype(int); d['date']=pd.to_datetime(dict(year=d.year,month=d.month,day=1))+pd.offsets.MonthEnd(0); d['value']=pd.to_numeric(d.value,errors='coerce'); return d[['series_id','date','value']]
h=load_bls(DATA/'ei_harmonized_import.tsv'); b=load_bls(DATA/'ei_bea_import.tsv'); uni=pd.read_csv(OUT/'primary_universe.csv'); sids=uni.series_id.tolist(); labels=uni.set_index('series_id').series_name.to_dict(); hp=h[h.series_id.isin(sids+['EIUIP27'])].pivot(index='date',columns='series_id',values='value').sort_index(); bea=b[b.series_id.eq('EIUIREXFUELS')].pivot(index='date',columns='series_id',values='value').sort_index(); g=pd.read_excel(DATA/'gscpi_conv/gscpi_data.xlsx',sheet_name='GSCPI Monthly Data').iloc[:,0:2]; g.columns=['date','g']; g['date']=pd.to_datetime(g.date,errors='coerce'); g['g']=pd.to_numeric(g.g,errors='coerce'); g=g.dropna().set_index('date'); g.index=g.index+pd.offsets.MonthEnd(0)
idx=pd.date_range(max(hp.index.min(),bea.index.min(),g.index.min()),min(hp.index.max(),bea.index.max(),g.index.max()),freq='ME'); c=pd.DataFrame(index=idx); c['exfuel']=bea.EIUIREXFUELS.reindex(idx); c['fuel']=hp.EIUIP27.reindex(idx); c['g']=g.g.reindex(idx); c['gclip']=c.g.clip(-3,3); c['gtanh']=np.tanh(c.g/2); c['dg1']=c.g.diff(); c['dg3']=c.g.diff(3)
for col in ['exfuel','fuel']:
 lp=np.log(c[col]); c[f'{col}_m1']=100*lp.diff(); c[f'{col}_m3']=100*lp.diff(3); c[f'{col}_m12']=100*lp.diff(12)
for root in ['g','gclip','gtanh','dg1']:
 for j in range(7): c[f'{root}_l{j}']=c[root].shift(j)
 c[f'{root}_mean3']=c[root].rolling(3).mean(); c[f'{root}_chg3']=c[root]-c[root].shift(3)
c['month_sin']=np.sin(2*np.pi*c.index.month/12); c['month_cos']=np.cos(2*np.pi*c.index.month/12)
BASE=['own_m1_l0','own_m1_l1','own_m1_l2','own_m1_l3','own_m1_l6','own_m1_l12','own_m3','own_m12','exfuel_m1','exfuel_m3','exfuel_m12','fuel_m1','fuel_m3','fuel_m12','month_sin','month_cos']
VAR={
 'level_full':BASE+[f'g_l{j}' for j in range(7)]+['g_mean3','g_chg3'],
 'level_short':BASE+[f'g_l{j}' for j in range(4)],
 'change_short':BASE+[f'dg1_l{j}' for j in range(4)]+['dg3'],
 'clipped_full':BASE+[f'gclip_l{j}' for j in range(7)]+['gclip_mean3','gclip_chg3'],
 'tanh_full':BASE+[f'gtanh_l{j}' for j in range(7)]+['gtanh_mean3','gtanh_chg3']
}
ALPHAS=np.array([.01,.1,1,10,100,1000.]); hh=3

def mk(sid):
 d=c.copy(); lp=np.log(hp[sid].reindex(d.index)); m=100*lp.diff(); d['own_m1_l0']=m
 for j in [1,2,3,6,12]: d[f'own_m1_l{j}']=m.shift(j)
 d['own_m3']=100*lp.diff(3); d['own_m12']=100*lp.diff(12); d['target']=100*(lp.shift(-hh)-lp); d['target_end']=d.index+pd.offsets.MonthEnd(hh); return d

def tune(d,F):
 dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]'); X=d[F].to_numpy(float); y=d.target.to_numpy(float); ok=np.isfinite(X).all(1)&np.isfinite(y); ss=np.zeros(len(ALPHAS)); nn=np.zeros(len(ALPHAS))
 for vy in range(2012,2017):
  vs=np.datetime64(f'{vy}-01-31'); ve=np.datetime64(f'{vy}-12-31'); tr=ok&(tend<vs); va=ok&(dates>=vs)&(dates<=ve)&(tend<=np.datetime64('2016-12-31'))
  if tr.sum()<60 or va.sum()<4: continue
  sc=StandardScaler(); xt=sc.fit_transform(X[tr]); xv=sc.transform(X[va]);
  for k,a in enumerate(ALPHAS):
   m=Ridge(alpha=a).fit(xt,y[tr]); e=y[va]-m.predict(xv); ss[k]+=e@e; nn[k]+=len(e)
 return float(ALPHAS[np.argmin(np.where(nn>0,ss/nn,np.inf))])

def fitpred(X,y,tr,xi,a):
 sc=StandardScaler(); xt=sc.fit_transform(X[tr]); xv=sc.transform(X[[xi]]); return float(Ridge(alpha=a).fit(xt,y[tr]).predict(xv)[0])

recs=[]
for sid in sids:
 d=mk(sid); dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]'); y=d.target.to_numpy(float)
 for name,F in [('baseline',BASE)]+list(VAR.items()):
  a=tune(d,F); X=d[F].to_numpy(float); ok=np.isfinite(X).all(1)&np.isfinite(y); origins=np.flatnonzero(ok&(dates>=np.datetime64('2017-01-31'))&(tend<=dates.max()))
  for i in origins:
   tr=ok&(tend<=dates[i]);
   if tr.sum()<100: continue
   recs.append({'series_id':sid,'series_name':labels[sid],'origin':pd.Timestamp(dates[i]),'target_end':pd.Timestamp(tend[i]),'model':name,'pred':fitpred(X,y,tr,i,a),'actual':float(y[i]),'alpha':a})

f=pd.DataFrame(recs); f.to_csv(OUT/'gscpi_variant_forecasts_h3.csv',index=False)
# align every variant to baseline per cell
base=f[f.model.eq('baseline')][['series_id','origin','actual','pred']].rename(columns={'pred':'base'}); z=f.merge(base,on=['series_id','origin','actual'],how='inner'); z['se_model']=(z.actual-z.pred)**2; z['se_base']=(z.actual-z.base)**2
rows=[]
for sample,mask in [('development_2017_2022',z.origin<pd.Timestamp('2023-01-31')),('validation_2023_2026',z.origin>=pd.Timestamp('2023-01-31'))]:
 q=z[mask]
 for m,r in q.groupby('model'):
  if m=='baseline': continue
  rr=np.sqrt(r.se_model.mean())/np.sqrt(r.se_base.mean()); rows.append({'sample':sample,'model':m,'n':len(r),'rrmse':rr,'rmse_reduction_pct':100*(1-rr)})
res=pd.DataFrame(rows); res.to_csv(OUT/'gscpi_variant_metrics_h3.csv',index=False)
# choose best on development, then block bootstrap validation for that choice
best=res[res['sample'].eq('development_2017_2022')].sort_values('rrmse').iloc[0].model
q=z[(z.model.eq(best))&(z.origin>=pd.Timestamp('2023-01-31'))].copy(); ag=q.groupby('origin').agg(sm=('se_model','sum'),sb=('se_base','sum'),n=('se_base','size')).sort_index(); N=len(ag); sm=ag.sm.values; sb=ag.sb.values; nn=ag.n.values; point=np.sqrt(sm.sum()/nn.sum())/np.sqrt(sb.sum()/nn.sum()); rng=np.random.default_rng(20260824); vals=[]; block=6
for _ in range(5000):
 ids=[]
 while len(ids)<N:
  s=int(rng.integers(0,max(1,N-block+1))); ids.extend(range(s,min(N,s+block)))
 ids=np.array(ids[:N]); vals.append(np.sqrt(sm[ids].sum()/nn[ids].sum())/np.sqrt(sb[ids].sum()/nn[ids].sum()))
lo,hi=np.quantile(vals,[.025,.975]); summ={'selected_on_development':best,'validation_rrmse':float(point),'validation_rmse_reduction_pct':float(100*(1-point)),'validation_95ci':[float(lo),float(hi)],'prob_rrmse_lt_1':float(np.mean(np.array(vals)<1))}
json.dump(summ,open(OUT/'gscpi_variant_validation_summary.json','w'),indent=2)
print(res.sort_values(['sample','rrmse']).to_string(index=False)); print('\nSELECTED + VALIDATION\n',json.dumps(summ,indent=2))
