#!/usr/bin/env python3
"""Granular logistics-network forecasting study.

Primary confirmatory test: Does baseline + GSCPI + granular Census network data
(M3) beat baseline + GSCPI (M1) for 3-month-ahead BLS HS import-price changes?

This script is designed to run after the official Census bundle made by
`download_census_porths.py` is available. It deliberately imposes a one-month
extra lag on Census reference months relative to the latest known BLS price
reference month, to approximate the later trade-data publication schedule.

The retrospective Census history is revised/current-vintage data, so results
remain pseudo-real-time; the script labels them accordingly.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HS2 = [29,30,38,39,40,62,64,73,74,76,82,83,84,85,87,90,95,96]
NUM = ["GEN_VAL_MO","AIR_VAL_MO","AIR_WGT_MO","VES_VAL_MO","VES_WGT_MO","CNT_VAL_MO","CNT_WGT_MO"]
ALPHAS=np.array([0.01,0.1,1,10,100,1000.],float)
HORIZONS=[1,3,6]
TEST_START=pd.Timestamp('2017-01-31') # price reference month s; issued during s+1
COVID_START=pd.Timestamp('2020-03-31'); POST_START=pd.Timestamp('2023-01-31')

BASE=["own_m1_l0","own_m1_l1","own_m1_l2","own_m1_l3","own_m1_l6","own_m1_l12","own_m3","own_m12",
      "exfuel_m1","exfuel_m3","exfuel_m12","fuel_m1","fuel_m3","fuel_m12","month_sin","month_cos"]
GSC=[f"gscpi_l{j}" for j in range(7)]+["gscpi_mean3","gscpi_chg3"]
NET_BASIC=[
    "net_gen_val_yoy","net_ves_val_yoy","net_ves_wgt_yoy","net_cnt_val_yoy","net_cnt_wgt_yoy",
    "net_ocean_share","net_air_share","net_container_share_total","net_container_share_vessel",
    "net_vessel_value_per_kg_yoy","net_container_value_per_kg_yoy",
    "net_port_hhi","net_top_port_share","net_port_turnover12",
    "net_country_hhi","net_top_country_share","net_country_turnover12"
]
NET_ENV=[
    "net_port_env_ves_wgt_yoy","net_port_env_cnt_wgt_yoy","net_port_env_coverage",
    "net_country_env_ves_wgt_yoy","net_country_env_coverage"
]
NET=NET_BASIC+NET_ENV
MODELS={"M0_baseline":BASE,"M1_gscpi":BASE+GSC,"M2_network":BASE+NET,"M3_gscpi_network":BASE+GSC+NET}


def load_bls(path):
    d=pd.read_csv(path,sep='\t'); d.columns=d.columns.str.strip()
    d['series_id']=d.series_id.astype(str).str.strip(); d['period']=d.period.astype(str).str.strip()
    d=d[d.period.str.fullmatch(r'M(0[1-9]|1[0-2])',na=False)].copy(); d['month']=d.period.str[1:].astype(int)
    d['date']=pd.to_datetime(dict(year=d.year,month=d.month,day=1))+pd.offsets.MonthEnd(0)
    d['value']=pd.to_numeric(d.value,errors='coerce')
    return d[['series_id','date','value']]


def open_census_members(bundle: Path):
    z=zipfile.ZipFile(bundle)
    members=[m for m in z.namelist() if '/raw/' in m and m.endswith('.csv.gz')]
    if not members: raise RuntimeError('No /raw/*.csv.gz members found in Census bundle')
    return z,members


def read_and_aggregate_census(bundle: Path, cache: Path):
    """Streaming-ish aggregation. Cache intermediate product/month, product/port/month,
    and product/country/month tables so model reruns do not reread raw downloads."""
    cache.mkdir(parents=True,exist_ok=True)
    paths={k:cache/f'{k}.parquet' for k in ['prod','port','country']}
    if all(p.exists() for p in paths.values()):
        return tuple(pd.read_parquet(paths[k]) for k in ['prod','port','country'])
    pkl={k:cache/f'{k}.pkl' for k in ['prod','port','country']}
    if all(p.exists() for p in pkl.values()):
        return tuple(pd.read_pickle(pkl[k]) for k in ['prod','port','country'])
    z,members=open_census_members(bundle)
    prods=[]; ports=[]; countries=[]
    use=['I_COMMODITY','CTY_CODE','PORT','YEAR','MONTH']+NUM
    for i,m in enumerate(members,1):
        with z.open(m) as raw, gzip.GzipFile(fileobj=raw) as gz:
            d=pd.read_csv(gz,usecols=lambda c:c in use,dtype={'I_COMMODITY':'string','CTY_CODE':'string','PORT':'string','YEAR':'string','MONTH':'string'},low_memory=False)
        d['hs2']=pd.to_numeric(d['I_COMMODITY'].astype(str).str[:2],errors='coerce').astype('Int64')
        d=d[d.hs2.isin(HS2)].copy()
        for c in NUM: d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0.0)
        d['date']=pd.to_datetime(d['YEAR'].astype(str)+'-'+d['MONTH'].astype(str).str.zfill(2)+'-01',errors='coerce')+pd.offsets.MonthEnd(0)
        d=d.dropna(subset=['date','hs2'])
        prods.append(d.groupby(['hs2','date'],as_index=False)[NUM].sum())
        ports.append(d.groupby(['hs2','date','PORT'],as_index=False)[['GEN_VAL_MO','VES_VAL_MO','VES_WGT_MO','CNT_VAL_MO','CNT_WGT_MO']].sum())
        countries.append(d.groupby(['hs2','date','CTY_CODE'],as_index=False)[['GEN_VAL_MO','VES_VAL_MO','VES_WGT_MO','CNT_VAL_MO','CNT_WGT_MO']].sum())
        if i%25==0: print(f'  read {i}/{len(members)} Census files',flush=True)
    z.close()
    prod=pd.concat(prods,ignore_index=True).groupby(['hs2','date'],as_index=False)[NUM].sum()
    port=pd.concat(ports,ignore_index=True).groupby(['hs2','date','PORT'],as_index=False)[['GEN_VAL_MO','VES_VAL_MO','VES_WGT_MO','CNT_VAL_MO','CNT_WGT_MO']].sum()
    country=pd.concat(countries,ignore_index=True).groupby(['hs2','date','CTY_CODE'],as_index=False)[['GEN_VAL_MO','VES_VAL_MO','VES_WGT_MO','CNT_VAL_MO','CNT_WGT_MO']].sum()
    for k,x in [('prod',prod),('port',port),('country',country)]:
        try: x.to_parquet(paths[k],index=False)
        except Exception: x.to_pickle(cache/f'{k}.pkl')
    return prod,port,country


def concentration_features(tbl, entity, value_col, prefix):
    x=tbl[['hs2','date',entity,value_col]].copy(); x[value_col]=x[value_col].clip(lower=0)
    total=x.groupby(['hs2','date'])[value_col].transform('sum')
    x['share']=np.where(total>0,x[value_col]/total,0.0)
    agg=x.groupby(['hs2','date']).agg(**{f'{prefix}_hhi':('share',lambda s:float(np.square(s).sum())),f'{prefix}_top':('share','max')}).reset_index()
    lag=x[['hs2','date',entity,'share']].copy(); lag['date']=lag['date']+pd.offsets.MonthEnd(12); lag=lag.rename(columns={'share':'share_lag'})
    m=x[['hs2','date',entity,'share']].merge(lag,on=['hs2','date',entity],how='outer'); m[['share','share_lag']]=m[['share','share_lag']].fillna(0.0)
    turn=m.groupby(['hs2','date']).apply(lambda z:0.5*np.abs(z.share-z.share_lag).sum(),include_groups=False).rename(f'{prefix}_turnover12').reset_index()
    return agg.merge(turn,on=['hs2','date'],how='left')


def env_exposure_features(entity_tbl, entity, weight_value, prefix, idx):
    """Product exposure to leave-one-product-out node flow conditions.
    Exposure weights are frozen using calendar-2010 target-product import shares.
    Node shocks are YoY log changes in physical vessel/container weight among the
    OTHER 17 target categories, reducing mechanical own-product feedback."""
    vals=['VES_WGT_MO','CNT_WGT_MO']
    total=entity_tbl.groupby(['date',entity],as_index=False)[vals].sum()
    total_piv={v:total.pivot(index='date',columns=entity,values=v).reindex(idx).fillna(0.0) for v in vals}
    wsrc=entity_tbl[entity_tbl.date.dt.year.eq(2010)].groupby(['hs2',entity],as_index=False)[weight_value].sum()
    rows=[]
    for hs in HS2:
        w=wsrc[wsrc.hs2.eq(hs)].set_index(entity)[weight_value].clip(lower=0)
        if w.sum()<=0:
            # fallback to general import exposure if vessel-weighted map is absent
            fallback=entity_tbl[(entity_tbl.hs2.eq(hs))&(entity_tbl.date.dt.year.eq(2010))].groupby(entity)['GEN_VAL_MO'].sum().clip(lower=0)
            w=fallback
        if w.sum()<=0: continue
        w=(w/w.sum()); cols=[c for c in w.index if c in total_piv['VES_WGT_MO'].columns]
        if not cols: continue
        w=w.loc[cols]
        ph=entity_tbl[entity_tbl.hs2.eq(hs)]
        out=pd.DataFrame(index=idx)
        cover_components=[]
        for v,short in [('VES_WGT_MO','ves_wgt'),('CNT_WGT_MO','cnt_wgt')]:
            tot=total_piv[v][cols]
            pp=ph.pivot(index='date',columns=entity,values=v).reindex(index=idx,columns=cols).fillna(0.0)
            other=(tot-pp).clip(lower=0.0)
            yoy=100*(np.log1p(other)-np.log1p(other.shift(12)))
            finite=yoy.notna().astype(float)
            denom=finite.mul(w,axis=1).sum(axis=1)
            num=yoy.fillna(0).mul(w,axis=1).sum(axis=1)
            out[f'{prefix}_env_{short}_yoy']=np.where(denom>0,num/denom,np.nan)
            cover_components.append(denom)
        out[f'{prefix}_env_coverage']=pd.concat(cover_components,axis=1).min(axis=1)
        out['hs2']=hs; out['date']=out.index; rows.append(out.reset_index(drop=True))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['hs2','date'])


def build_network_features(prod,port,country):
    idx=pd.date_range(prod.date.min(),prod.date.max(),freq='ME')
    f=prod.copy().sort_values(['hs2','date'])
    # Ensure full monthly grid for each product.
    grid=pd.MultiIndex.from_product([HS2,idx],names=['hs2','date']).to_frame(index=False)
    f=grid.merge(f,on=['hs2','date'],how='left'); f[NUM]=f[NUM].fillna(0.0); f=f.sort_values(['hs2','date'])
    g=f.groupby('hs2',group_keys=False)
    for c,nm in [('GEN_VAL_MO','gen_val'),('VES_VAL_MO','ves_val'),('VES_WGT_MO','ves_wgt'),('CNT_VAL_MO','cnt_val'),('CNT_WGT_MO','cnt_wgt')]:
        f[f'net_{nm}_yoy']=100*(np.log1p(f[c])-g[c].transform(lambda s:np.log1p(s).shift(12)))
    f['net_ocean_share']=np.where(f.GEN_VAL_MO>0,f.VES_VAL_MO/f.GEN_VAL_MO,0.0)
    f['net_air_share']=np.where(f.GEN_VAL_MO>0,f.AIR_VAL_MO/f.GEN_VAL_MO,0.0)
    f['net_container_share_total']=np.where(f.GEN_VAL_MO>0,f.CNT_VAL_MO/f.GEN_VAL_MO,0.0)
    f['net_container_share_vessel']=np.where(f.VES_VAL_MO>0,f.CNT_VAL_MO/f.VES_VAL_MO,0.0)
    vpkg=np.where(f.VES_WGT_MO>0,f.VES_VAL_MO/f.VES_WGT_MO,np.nan); cpkg=np.where(f.CNT_WGT_MO>0,f.CNT_VAL_MO/f.CNT_WGT_MO,np.nan)
    f['_vpkg']=vpkg; f['_cpkg']=cpkg
    f['net_vessel_value_per_kg_yoy']=100*(np.log(f['_vpkg'])-g['_vpkg'].transform(lambda s:np.log(s).shift(12)))
    f['net_container_value_per_kg_yoy']=100*(np.log(f['_cpkg'])-g['_cpkg'].transform(lambda s:np.log(s).shift(12)))
    pf=concentration_features(port,'PORT','VES_VAL_MO','net_port').rename(columns={'net_port_top':'net_top_port_share'})
    cf=concentration_features(country,'CTY_CODE','GEN_VAL_MO','net_country').rename(columns={'net_country_top':'net_top_country_share'})
    pf=pf.rename(columns={'net_port_turnover12':'net_port_turnover12'})
    cf=cf.rename(columns={'net_country_turnover12':'net_country_turnover12'})
    pe=env_exposure_features(port,'PORT','VES_VAL_MO','net_port',idx)
    ce=env_exposure_features(country,'CTY_CODE','GEN_VAL_MO','net_country',idx)
    # Country environment only uses vessel weight primary + coverage; retain but drop cnt counterpart from primary list.
    out=f[['hs2','date']+NET_BASIC[:11]].merge(pf,on=['hs2','date'],how='left').merge(cf,on=['hs2','date'],how='left').merge(pe,on=['hs2','date'],how='left').merge(ce,on=['hs2','date'],how='left')
    # Harmonize env names expected by NET_ENV
    # generated: net_port_env_ves_wgt_yoy, net_port_env_cnt_wgt_yoy, net_port_env_coverage etc.
    # generated country counterpart includes cnt feature but it is exploratory/not in primary feature list.
    return out.sort_values(['hs2','date'])


def ridge_predict(Xtr,ytr,Xp,alpha):
    sc=StandardScaler(); Xt=sc.fit_transform(Xtr); xp=sc.transform(Xp); m=Ridge(alpha=alpha).fit(Xt,ytr); return m.predict(xp)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--census-bundle',type=Path,required=True)
    ap.add_argument('--bls-harmonized',type=Path,default=Path('/mnt/data/ei_harmonized_import.tsv'))
    ap.add_argument('--bls-bea',type=Path,default=Path('/mnt/data/ei_bea_import.tsv'))
    ap.add_argument('--bls-meta',type=Path,default=Path('/mnt/data/ei_series.tsv'))
    ap.add_argument('--gscpi',type=Path,default=Path('/mnt/data/gscpi_conv/gscpi_data.xlsx'))
    ap.add_argument('--out',type=Path,default=Path('/mnt/data/granular_network_study_outputs'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    print('1/6 Reading + aggregating Census network data...',flush=True)
    prod,port,country=read_and_aggregate_census(args.census_bundle,args.out/'cache')
    print('2/6 Building network features...',flush=True)
    net=build_network_features(prod,port,country); net.to_csv(args.out/'network_features_monthly.csv.gz',index=False,compression='gzip')

    print('3/6 Loading BLS + GSCPI...',flush=True)
    h=load_bls(args.bls_harmonized); b=load_bls(args.bls_bea)
    meta=pd.read_csv(args.bls_meta,sep='\t'); meta.columns=meta.columns.str.strip(); meta['series_id']=meta.series_id.astype(str).str.strip()
    sids=[f'EIUIP{x:02d}' for x in HS2]; labels=meta[meta.series_id.isin(sids)].set_index('series_id').series_name.to_dict()
    hp=h[h.series_id.isin(sids+['EIUIP27'])].pivot(index='date',columns='series_id',values='value').sort_index()
    bea=b[b.series_id.eq('EIUIREXFUELS')].pivot(index='date',columns='series_id',values='value').sort_index()
    gs=pd.read_excel(args.gscpi,sheet_name='GSCPI Monthly Data').iloc[:,:2]; gs.columns=['date','gscpi']; gs['date']=pd.to_datetime(gs.date,errors='coerce')+pd.offsets.MonthEnd(0); gs['gscpi']=pd.to_numeric(gs.gscpi,errors='coerce'); gs=gs.dropna().set_index('date').sort_index()
    idx=pd.date_range(max(hp.index.min(),bea.index.min(),gs.index.min()),min(hp.index.max(),bea.index.max(),gs.index.max()),freq='ME')
    common=pd.DataFrame(index=idx); common['exfuel']=bea['EIUIREXFUELS'].reindex(idx); common['fuel']=hp['EIUIP27'].reindex(idx); common['gscpi']=gs.gscpi.reindex(idx)
    for col in ['exfuel','fuel']:
        lp=np.log(common[col]); common[f'{col}_m1']=100*lp.diff(); common[f'{col}_m3']=100*lp.diff(3); common[f'{col}_m12']=100*lp.diff(12)
    for j in range(7): common[f'gscpi_l{j}']=common.gscpi.shift(j)
    common['gscpi_mean3']=common.gscpi.rolling(3).mean(); common['gscpi_chg3']=common.gscpi-common.gscpi.shift(3)
    common['month_sin']=np.sin(2*np.pi*common.index.month/12); common['month_cos']=np.cos(2*np.pi*common.index.month/12)

    net_by_hs={hs:net[net.hs2.eq(hs)].set_index('date').drop(columns='hs2').sort_index() for hs in HS2}
    def make_df(sid,hh):
        hs=int(sid[-2:]); d=common.copy(); lp=np.log(hp[sid].reindex(d.index)); m1=100*lp.diff(); d['own_m1_l0']=m1
        for j in [1,2,3,6,12]: d[f'own_m1_l{j}']=m1.shift(j)
        d['own_m3']=100*lp.diff(3); d['own_m12']=100*lp.diff(12); d['target']=100*(lp.shift(-hh)-lp); d['target_end']=d.index+pd.offsets.MonthEnd(hh)
        # Publication-aware alignment: when BLS reference month s is known in s+1, use Census only through s-1.
        n=net_by_hs[hs].reindex(d.index).shift(1)
        d=d.join(n)
        return d

    def tune(d,features):
        dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]'); X=d[features].to_numpy(float); y=d.target.to_numpy(float); valid=np.isfinite(X).all(1)&np.isfinite(y)
        mse=np.zeros(len(ALPHAS)); n=np.zeros(len(ALPHAS))
        # Pre-test CV only. Network history begins around 2011; use 2014-2016 folds.
        for vy in [2014,2015,2016]:
            vs=np.datetime64(f'{vy}-01-31'); ve=np.datetime64(f'{vy}-12-31'); tr=valid&(tend<vs); va=valid&(dates>=vs)&(dates<=ve)&(tend<=np.datetime64('2016-12-31'))
            if tr.sum()<24 or va.sum()<4: continue
            sc=StandardScaler(); Xt=sc.fit_transform(X[tr]); Xv=sc.transform(X[va]); yt=y[tr]; yv=y[va]
            for k,a in enumerate(ALPHAS):
                e=yv-Ridge(alpha=a).fit(Xt,yt).predict(Xv); mse[k]+=e@e; n[k]+=len(e)
        score=np.where(n>0,mse/n,np.inf); return float(ALPHAS[np.argmin(score)]),score

    print('4/6 Running rolling forecasts...',flush=True)
    records=[]; alpha_rows=[]
    for hh in HORIZONS:
        for sid in sids:
            d=make_df(sid,hh); dates=d.index.values.astype('datetime64[D]'); tend=d.target_end.values.astype('datetime64[D]')
            # Fair comparison: define eligible origins by full M3 feature availability, then run every model on same origins.
            full_features=MODELS['M3_gscpi_network']; Xfull=d[full_features].to_numpy(float); y=d.target.to_numpy(float)
            origin_eligible=(dates>=np.datetime64(str(TEST_START.date())))&np.isfinite(Xfull).all(1)&np.isfinite(y)&(tend<=dates.max())
            for model,features in MODELS.items():
                alpha,scores=tune(d,features); alpha_rows.append({'series_id':sid,'horizon':hh,'model':model,'alpha':alpha,**{f'cv_mse_{a}':v for a,v in zip(ALPHAS,scores)}})
                X=d[features].to_numpy(float); valid=np.isfinite(X).all(1)&np.isfinite(y)
                for oi in np.flatnonzero(origin_eligible):
                    # target_end <= current BLS reference month => target was already known at this forecast issue
                    tr=valid&(tend<=dates[oi])
                    if tr.sum()<36: continue
                    pred=ridge_predict(X[tr],y[tr],X[[oi]],alpha)[0]
                    records.append({'series_id':sid,'series_name':labels.get(sid,sid),'price_reference_month':pd.Timestamp(dates[oi]),
                                    'forecast_issue_proxy':pd.Timestamp(dates[oi])+pd.offsets.MonthEnd(1),'target_end':pd.Timestamp(tend[oi]),'horizon':hh,
                                    'model':model,'prediction':float(pred),'actual':float(y[oi]),'n_train':int(tr.sum()),'alpha':alpha})
            print('  done',hh,sid,flush=True)
    fc=pd.DataFrame(records); fc.to_csv(args.out/'rolling_forecasts.csv.gz',index=False,compression='gzip'); pd.DataFrame(alpha_rows).to_csv(args.out/'ridge_alpha_selection.csv',index=False)

    print('5/6 Scoring + block bootstrap...',flush=True)
    piv=fc.pivot_table(index=['series_id','series_name','price_reference_month','forecast_issue_proxy','target_end','horizon','actual'],columns='model',values='prediction').reset_index().dropna()
    for m in MODELS: piv[f'se_{m}']=(piv.actual-piv[m])**2
    piv.to_csv(args.out/'paired_forecasts.csv.gz',index=False,compression='gzip')
    samples={
        'full':np.ones(len(piv),dtype=bool),
        'ex_covid':(piv.target_end<COVID_START)|(piv.price_reference_month>=POST_START),
        'pre_covid':piv.target_end<COVID_START,
        'covid':(piv.target_end>=COVID_START)&(piv.price_reference_month<POST_START),
        'post_covid':piv.price_reference_month>=POST_START
    }
    metric=[]
    for s,mask in samples.items():
        z=piv[mask]
        for hh in HORIZONS:
            zh=z[z.horizon.eq(hh)]
            for m in MODELS:
                rm=np.sqrt(zh[f'se_{m}'].mean()); metric.append({'sample':s,'horizon':hh,'model':m,'n':len(zh),'rmse':rm})
    metric=pd.DataFrame(metric); metric.to_csv(args.out/'aggregate_metrics.csv',index=False)

    def boot_compare(hh,sample,num='M3_gscpi_network',den='M1_gscpi',B=5000,block=6,seed=20260824):
        z=piv[(piv.horizon.eq(hh))&samples[sample]].copy(); ag=z.groupby('price_reference_month').agg(sn=(f'se_{num}','sum'),sd=(f'se_{den}','sum'),n=('actual','size')).sort_index(); N=len(ag)
        sn,sd,nn=ag.sn.values,ag.sd.values,ag.n.values; point=np.sqrt(sn.sum()/nn.sum())/np.sqrt(sd.sum()/nn.sum()); rng=np.random.default_rng(seed+hh+len(sample)); vals=np.empty(B); maxs=max(1,N-block+1)
        for b in range(B):
            ids=[]
            while len(ids)<N:
                st=int(rng.integers(0,maxs)); ids.extend(range(st,min(st+block,N)))
            ids=np.asarray(ids[:N]); vals[b]=np.sqrt(sn[ids].sum()/nn[ids].sum())/np.sqrt(sd[ids].sum()/nn[ids].sum())
        lo,hi=np.quantile(vals,[.025,.975]); return {'horizon':hh,'sample':sample,'numerator':num,'denominator':den,'n_months':N,'n_cells':len(z),'rrmse':point,'rmse_reduction_pct':100*(1-point),'ci_low':lo,'ci_high':hi,'prob_rrmse_lt_1':float((vals<1).mean())}
    boots=[]
    for hh in HORIZONS:
        for s in samples:
            boots.append(boot_compare(hh,s,'M3_gscpi_network','M1_gscpi'))
            boots.append(boot_compare(hh,s,'M2_network','M0_baseline'))
    boots=pd.DataFrame(boots); boots.to_csv(args.out/'bootstrap_comparisons.csv',index=False)

    cats=[]
    for (hh,sid,name),z in piv.groupby(['horizon','series_id','series_name']):
        for num,den,label in [('M3_gscpi_network','M1_gscpi','incremental_network_over_gscpi'),('M2_network','M0_baseline','network_over_baseline')]:
            rn=np.sqrt(z[f'se_{num}'].mean()); rd=np.sqrt(z[f'se_{den}'].mean()); cats.append({'horizon':hh,'series_id':sid,'series_name':name,'comparison':label,'n':len(z),'rrmse':rn/rd,'rmse_reduction_pct':100*(1-rn/rd)})
    pd.DataFrame(cats).to_csv(args.out/'category_metrics.csv',index=False)

    primary=boots[(boots.horizon.eq(3))&(boots['sample'].eq('full'))&(boots.numerator.eq('M3_gscpi_network'))].iloc[0]
    exc=boots[(boots.horizon.eq(3))&(boots['sample'].eq('ex_covid'))&(boots.numerator.eq('M3_gscpi_network'))].iloc[0]
    passed=bool(primary.rrmse<=0.95 and primary.ci_high<1.0 and exc.rrmse<=0.98)
    result={
      'study_version':'granular-network-v1-2026-08-24','status':'PASS' if passed else 'FAIL','primary_comparison':'M3_gscpi_network vs M1_gscpi',
      'primary_horizon_months':3,'full_rrmse':float(primary.rrmse),'full_rmse_reduction_pct':float(primary.rmse_reduction_pct),'full_95ci':[float(primary.ci_low),float(primary.ci_high)],
      'ex_covid_rrmse':float(exc.rrmse),'ex_covid_rmse_reduction_pct':float(exc.rmse_reduction_pct),'success_rule_met':passed,
      'data_status':'pseudo-real-time retrospective Census history with conservative publication lag; prospective snapshots required for true real-time validation'
    }
    (args.out/'decision_summary.json').write_text(json.dumps(result,indent=2))
    print('6/6 DONE\n'+json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
