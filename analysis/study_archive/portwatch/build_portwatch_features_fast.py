from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path('/mnt/data')
OUT=ROOT/'portwatch_ais_study_outputs'
orig=pd.read_csv(ROOT/'portwatch_extracted/portwatch_monthly_bundle/origin_country_monthly.csv')
usa=pd.read_csv(ROOT/'portwatch_extracted/portwatch_monthly_bundle/usa_port_monthly.csv')
country=pd.read_csv(OUT/'origin_exposure_weights.csv',dtype={'I_COMMODITY':str})
ports=pd.read_csv(OUT/'census_to_portwatch_port_map.csv',dtype={'I_COMMODITY':str})
ports=ports[ports.pw_portname.notna()][['I_COMMODITY','pw_portname','weight']]
routes=pd.read_csv(OUT/'matched_route_exposure_weights.csv',dtype={'I_COMMODITY':str})
coverage=pd.read_csv(OUT/'exposure_coverage.csv',dtype={'hs':str})
all_months=pd.period_range('2019-01','2026-08',freq='M')

def make_shocks(df,key,metrics):
    parts=[]
    df=df.copy(); df['period']=pd.PeriodIndex.from_fields(year=df.year,month=df.month,freq='M')
    for kval,g in df.groupby(key):
        gg=g.set_index('period').reindex(all_months)
        z=pd.DataFrame({'period':all_months,key:kval})
        for m in metrics:
            s=pd.to_numeric(gg[m],errors='coerce')
            z['shock_'+m]=-(np.log1p(s.values)-np.log1p(s.shift(12).values))
        parts.append(z)
    return pd.concat(parts,ignore_index=True)

osh=make_shocks(orig,'ISO3',['export_container','export_cargo','portcalls_container'])
# only matched ports to reduce
usa=usa[usa.portname.isin(ports.pw_portname.unique())]
psh=make_shocks(usa,'portname',['import_container','import_cargo','portcalls_container']).rename(columns={'portname':'pw_portname'})
osh.to_csv(OUT/'origin_shocks.csv',index=False); psh.to_csv(OUT/'destination_port_shocks.csv',index=False)

# utility aggregate preserving fixed matched exposure magnitude

def agg_pressure(weights, shocks, key, shockcols, prefix):
    x=weights.merge(shocks,on=key,how='left')
    # fixed matched exposure per product
    matched=weights.groupby('I_COMMODITY').weight.sum().rename('matched').reset_index()
    outs=[]
    for scol,outname in shockcols.items():
        q=x[['I_COMMODITY','period','weight',scol]].copy()
        q['avail_w']=np.where(q[scol].notna(),q.weight,0.0)
        q['wx']=np.where(q[scol].notna(),q.weight*q[scol],0.0)
        g=q.groupby(['I_COMMODITY','period'],as_index=False).agg(avail_w=('avail_w','sum'),wx=('wx','sum'))
        g=g.merge(matched,on='I_COMMODITY',how='left')
        g[outname]=np.where(g.avail_w>=0.8*g.matched, g.matched*g.wx/g.avail_w, np.nan)
        outs.append(g[['I_COMMODITY','period',outname]])
    out=outs[0]
    for o in outs[1:]: out=out.merge(o,on=['I_COMMODITY','period'],how='outer')
    return out

org=agg_pressure(country[['I_COMMODITY','ISO3','weight']],osh,'ISO3',{
    'shock_export_container':'origin_export_container_pressure',
    'shock_export_cargo':'origin_export_cargo_pressure',
    'shock_portcalls_container':'origin_container_call_pressure'},'origin')

dst=agg_pressure(ports,psh,'pw_portname',{
    'shock_import_container':'destination_import_container_pressure',
    'shock_import_cargo':'destination_import_cargo_pressure',
    'shock_portcalls_container':'destination_container_call_pressure'},'destination')

# Route: expand routes by periods through joins, require both country and port shock observed.
rx=routes[['I_COMMODITY','ISO3','pw_portname','weight']].merge(osh,on='ISO3',how='left').merge(psh,on=['pw_portname','period'],how='left')
matched_route=routes.groupby('I_COMMODITY').weight.sum().rename('matched').reset_index()
route_out=[]
for oc,pc,oname in [
    ('shock_export_container','shock_import_container','route_container_pressure'),
    ('shock_export_cargo','shock_import_cargo','route_cargo_pressure')]:
    q=rx[['I_COMMODITY','period','weight',oc,pc]].copy()
    ok=q[oc].notna() & q[pc].notna()
    q['avail_w']=np.where(ok,q.weight,0.0)
    q['wx']=np.where(ok,q.weight*0.5*(q[oc]+q[pc]),0.0)
    g=q.groupby(['I_COMMODITY','period'],as_index=False).agg(avail_w=('avail_w','sum'),wx=('wx','sum'))
    g=g.merge(matched_route,on='I_COMMODITY',how='left')
    g[oname]=np.where(g.avail_w>=0.8*g.matched,g.matched*g.wx/g.avail_w,np.nan)
    route_out.append(g[['I_COMMODITY','period',oname]])
rte=route_out[0].merge(route_out[1],on=['I_COMMODITY','period'],how='outer')

feat=org.merge(dst,on=['I_COMMODITY','period'],how='outer').merge(rte,on=['I_COMMODITY','period'],how='outer')
feat=feat.rename(columns={'I_COMMODITY':'hs','period':'origin_period'})
feat=feat.merge(coverage[['hs','matched_route_exposure']],on='hs',how='left').rename(columns={'matched_route_exposure':'matched_route_exposure_coverage'})
BASE=[
'origin_export_container_pressure','origin_export_cargo_pressure','origin_container_call_pressure',
'destination_import_container_pressure','destination_import_cargo_pressure','destination_container_call_pressure',
'route_container_pressure','route_cargo_pressure']
feat=feat.sort_values(['hs','origin_period'])
for f in BASE:
    for lag in [1,2,3]: feat[f'{f}_lag{lag}']=feat.groupby('hs')[f].shift(lag)
preds=[]
for f in BASE: preds += [f]+[f'{f}_lag{i}' for i in [1,2,3]]
preds+=['matched_route_exposure_coverage']
feat.to_csv(OUT/'portwatch_product_month_features.csv',index=False)
av=feat.assign(complete=feat[preds].notna().all(axis=1)).groupby('origin_period').complete.mean().reset_index(name='product_complete_fraction')
av.to_csv(OUT/'feature_availability_by_month.csv',index=False)
print('features',feat.shape)
print('coverage summary')
print(coverage.to_string(index=False))
print('availability nonzero')
print(av[av.product_complete_fraction>0].head(12).to_string(index=False))
print('availability tail')
print(av.tail(12).to_string(index=False))
print('complete rows',feat[preds].notna().all(axis=1).sum(),'of',len(feat))
