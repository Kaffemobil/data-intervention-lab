from pathlib import Path
import json, numpy as np, pandas as pd
OUT=Path('/mnt/data/portwatch_ais_study_outputs')
p=pd.read_csv(OUT/'rolling_portwatch_forecasts.csv')
p['op']=pd.PeriodIndex(p.origin_period,freq='M')
CS=pd.Period('2020-03','M'); CE=pd.Period('2022-12','M')
rng=np.random.default_rng(20260825)

def boot(g,B=5000,L=6):
    a=(g.assign(e1=(g.actual-g.m1_prediction)**2,ep=(g.actual-g.pw_prediction)**2)
       .groupby('op',as_index=False).agg(sse1=('e1','sum'),ssep=('ep','sum'),n=('e1','size')).sort_values('op'))
    x1=a.sse1.to_numpy(); xp=a.ssep.to_numpy(); n=len(a)
    starts=np.arange(max(1,n-L+1))
    vals=np.empty(B)
    for b in range(B):
        idx=[]
        while len(idx)<n:
            s=int(rng.choice(starts)); idx.extend(range(s,min(s+L,n)))
        idx=np.array(idx[:n],dtype=int)
        vals[b]=np.sqrt(xp[idx].sum()/x1[idx].sum())
    return vals
rows=[]
for label,sub in [('full',p),('ex_covid',p[(p.op<CS)|(p.op>CE)]),('post_covid',p[p.op>CE])]:
    for h,g in sub.groupby('horizon'):
        v=boot(g)
        rows.append({'sample':label,'horizon':int(h),'draws':len(v),'mean_rrmse':v.mean(),'p2_5':np.quantile(v,.025),'p50':np.quantile(v,.5),'p97_5':np.quantile(v,.975),'prob_rrmse_lt_1':np.mean(v<1),'prob_rrmse_le_0_95':np.mean(v<=.95)})
b=pd.DataFrame(rows); b.to_csv(OUT/'bootstrap_rrmse.csv',index=False)
m=pd.read_csv(OUT/'aggregate_metrics.csv')
pr=m[(m['sample']=='full')&(m.horizon==3)].iloc[0]; ex=m[(m['sample']=='ex_covid')&(m.horizon==3)].iloc[0]; bb=b[(b['sample']=='full')&(b.horizon==3)].iloc[0]
passed=bool(pr.rrmse_portwatch<=.95 and bb.p97_5<1 and ex.rrmse_portwatch<=.98)
d={'passed':passed,'primary_h3_rrmse':float(pr.rrmse_portwatch),'primary_h3_rmse_reduction_pct':float(pr.rmse_reduction_pct),'primary_h3_bootstrap_95ci':[float(bb.p2_5),float(bb.p97_5)],'probability_rrmse_lt_1':float(bb.prob_rrmse_lt_1),'ex_covid_h3_rrmse':float(ex.rrmse_portwatch),'pw_vs_bias_rrmse_full_h3':float(pr.pw_vs_bias_rrmse),'success_rule':{'full_rrmse_max':.95,'full_95ci_upper_below':1.0,'ex_covid_rrmse_max':.98}}
(OUT/'stage_gate_decision.json').write_text(json.dumps(d,indent=2))
print(b.to_string(index=False)); print('\n',json.dumps(d,indent=2))
