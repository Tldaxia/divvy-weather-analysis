"""Discovery uses 2023 only; validation requires a frozen protocol written separately."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; RES=ROOT/'results'; FIG=ROOT/'figures'
plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'figure.dpi':140,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
COLORS={'casual':'#d8763a','member':'#247b92'}
CONTROLS='bs(temp, df=5, degree=3, lower_bound=-40, upper_bound=50) + AWND + np.log1p(SNOW) + np.log1p(SNWD) + C(month) + C(dow) + holiday'

def load(year,station='USW00094846',broad=False):
    t=pd.read_parquet(DATA/f"{'daily_broad' if broad else 'daily'}_{year}.parquet").pivot(index='date',columns='member_casual',values='trips').fillna(0)
    t.index=pd.to_datetime(t.index)
    t=t.reindex(pd.date_range(f'{year}-01-01',f'{year}-12-31'),fill_value=0)
    assert (t[['casual','member']]>0).all().all(), 'Unexpected missing or zero days'
    w=pd.read_csv(DATA/f'weather_{station}.csv',parse_dates=['date']).set_index('date')
    d=t.join(w)
    d['temp']=(d.TMAX+d.TMIN)/2
    d['month']=d.index.month;d['dow']=d.index.dayofweek
    hs=holidays.US(years=year)
    d['holiday']=[int(x.date() in hs) for x in d.index]
    d['wet']=(d.PRCP>=1).astype(int)
    d['total']=d.casual+d.member
    d['log_ratio']=np.log(d.casual/d.member)
    d['casual_share']=d.casual/d.total
    return d

def fit(d,outcome='log_ratio',treatment='wet',control=CONTROLS):
    return smf.ols(f'{outcome} ~ {treatment} + {control}',data=d,missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':7})

def estimate(m,term,label):
    lo,hi=m.conf_int().loc[term]
    return dict(label=label,n=int(m.nobs),beta=float(m.params[term]),se=float(m.bse[term]),p=float(m.pvalues[term]),
                ci_low=float(lo),ci_high=float(hi),percent=float(100*np.expm1(m.params[term])),
                percent_low=float(100*np.expm1(lo)),percent_high=float(100*np.expm1(hi)),r2=float(m.rsquared))

def discovery():
    d=load(2023)
    d.to_csv(RES/'discovery_daily.csv',index_label='date')
    q=pd.read_csv(DATA/'qc_2023.csv')
    summary=dict(year=2023,days=len(d),raw_trips=int(q.raw_rows.sum()),clean_trips=int(d.total.sum()),
                 missing_weather=d[['PRCP','TMAX','TMIN','AWND','SNOW','SNWD']].isna().sum().to_dict(),
                 wet_days=int(d.wet.sum()),dry_days=int((d.PRCP==0).sum()),
                 raw_wet_ratio=float(d.loc[d.wet==1,'casual'].sum()/d.loc[d.wet==1,'member'].sum()),
                 raw_dry_ratio=float(d.loc[d.PRCP==0,'casual'].sum()/d.loc[d.PRCP==0,'member'].sum()))
    rows=[]
    for outcome in ['log_ratio','np.log(casual)','np.log(member)']:
        m=fit(d,outcome)
        rows.append(estimate(m,'wet',outcome))
        (RES/f"discovery_model_{outcome.replace('np.log(','').replace(')','')}.txt").write_text(m.summary().as_text())
    pd.DataFrame(rows).to_csv(RES/'discovery_effects.csv',index=False)
    corr=d[['casual','member','total','casual_share','temp','PRCP','AWND','SNOW','SNWD']].corr(method='spearman')
    corr.to_csv(RES/'discovery_spearman.csv')
    summary['spearman_temp_total']=float(corr.loc['temp','total'])
    summary['effects']=rows
    (RES/'discovery_summary.json').write_text(json.dumps(summary,indent=2))
    inv=pd.read_csv(DATA/'station_inventory_2023.csv',dtype={'start_station_id':str})
    sd=pd.read_parquet(DATA/'station_daily_2023.parquet')
    stats=sd.groupby('start_station_id').agg(trips=('trips','sum'),active_days=('date','nunique'))
    groups=sd.groupby(['start_station_id','member_casual']).trips.sum().unstack(fill_value=0)
    stats=stats.join(groups)
    cohort=stats.query('trips >= 1000 and active_days >= 180 and casual >= 100 and member >= 100').index.tolist()
    (RES/'discovery_station_cohort.json').write_text(json.dumps(cohort,indent=2))
    stats.to_csv(RES/'discovery_station_summary.csv')
    summary['station_ids']=len(stats);summary['fixed_cohort']=len(cohort)
    summary['ids_multiple_names']=int((inv.groupby('start_station_id').start_station_name.nunique()>1).sum())
    summary['names_multiple_ids']=int((inv.groupby('start_station_name').start_station_id.nunique()>1).sum())
    (RES/'discovery_summary.json').write_text(json.dumps(summary,indent=2))
    # Time structure.
    fig,ax=plt.subplots(figsize=(11,3.7))
    for g in COLORS: ax.plot(d.index,d[g].rolling(7,center=True).mean(),label={'casual':'临时用户','member':'会员'}[g],color=COLORS[g])
    ax.set(title='2023 年每日骑行量：7 日移动平均',ylabel='行程数',xlabel='日期');ax.legend();fig.savefig(FIG/'01_daily_trends.png');plt.close(fig)
    h=pd.read_parquet(DATA/'hourly_2023.parquet');h['dow']=pd.to_datetime(h.date).dt.dayofweek;h['weekend']=h.dow>=5
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharey=True)
    for ax,we in zip(axes,[False,True]):
        sub=h[h.weekend==we];ndays=sub.date.nunique()
        for g in COLORS:
            curve=sub[sub.member_casual==g].groupby('hour').trips.sum()/ndays
            ax.plot(curve.index,curve.values,color=COLORS[g],label={'casual':'临时用户','member':'会员'}[g])
        ax.set(title='周末' if we else '工作日',xlabel='本地小时',ylabel='平均每小时行程数',xticks=range(0,24,3));ax.legend()
    fig.tight_layout();fig.savefig(FIG/'02_hourly_profiles.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set(xticks=range(len(corr)),yticks=range(len(corr)),xticklabels=corr.columns,yticklabels=corr.columns,title='2023 年日度 Spearman 相关性（未经混杂调整）')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    for i in range(len(corr)):
        for j in range(len(corr)): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax);fig.savefig(FIG/'03_correlations.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sc=axes[0].scatter(d.temp,d.total,c=d.PRCP.clip(upper=25),cmap='Blues',alpha=.7,s=18)
    axes[0].set(xlabel='日均温代理 (°C)',ylabel='每日总行程',title='温度、降水与需求');fig.colorbar(sc,ax=axes[0],label='降水 (mm，颜色上限25)')
    bins=pd.cut(d.PRCP,[-.01,0,1,5,15,1000],right=True)
    means=d.groupby(bins,observed=True).agg(share=('casual_share','mean'),days=('total','size'))
    means.to_csv(RES/'discovery_rain_bins.csv')
    axes[1].bar(range(len(means)),means.share,color='#247b92')
    axes[1].set(xticks=range(len(means)),xticklabels=['0','(0,1]','(1,5]','(5,15]','>15'],xlabel='日降水 (mm)',ylabel='临时用户行程占比',title='降水分组的原始差异（含季节混杂）')
    for i,n in enumerate(means.days): axes[1].text(i,means.share.iloc[i]+.005,f'n={n}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(FIG/'04_weather_patterns.png');plt.close(fig)
    top=inv.sort_values('trips',ascending=False).drop_duplicates('start_station_id').head(15).sort_values('trips')
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(top.start_station_name,top.trips,color='#247b92');ax.set(title='2023 年出发行程最多的 15 个站点',xlabel='有效出发行程');fig.savefig(FIG/'05_top_stations.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.8));ax.bar(q.month,q.missing_start_station_clean/q.clean_rows*100,color='#d8763a')
    ax.set(title='站点 ID 缺失是数据质量问题，不能当作零需求',xlabel='2023 年月份',ylabel='有效行程起点 ID 缺失 (%)',xticks=range(1,13));fig.savefig(FIG/'06_missing_stations.png');plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

def validation():
    protocol=ROOT/'refine-logs/protocol.json'
    assert protocol.exists(), 'Freeze protocol before looking at validation results'
    spec=json.loads(protocol.read_text())
    assert spec['primary_treatment']=='PRCP >= 1 mm' and spec['validation_year']==2024
    d=load(2024);train=load(2023)
    d.to_csv(RES/'validation_daily.csv',index_label='date')
    rows=[]
    primary=fit(d)
    rows.append(estimate(primary,'wet','2024 primary: city ratio'))
    (RES/'validation_primary_model.txt').write_text(primary.summary().as_text())
    for g in ['casual','member']:
        m=fit(d,f'np.log({g})');rows.append(estimate(m,'wet',f'2024 {g} demand'))
    # Exactly zero rain controls; omit trace/light-rain intermediate days.
    rows.append(estimate(fit(d[(d.PRCP==0)|(d.PRCP>=1)]),'wet','2024 zero-rain controls'))
    mw=d.copy();mid=load(2024,'USW00014819')
    for col in ['PRCP','TMAX','TMIN','AWND','temp','wet']: mw[col]=mid[col]
    rows.append(estimate(fit(mw),'wet','2024 Midway weather'))
    rows.append(estimate(fit(load(2024,broad=True)),'wet','2024 broad durations'))
    warm=d[(d.index.month>=4)&(d.index.month<=10)]
    rows.append(estimate(fit(warm),'wet','2024 Apr-Oct'))
    intense=d.copy();intense['wet']=(intense.PRCP>=5).astype(int)
    rows.append(estimate(fit(intense),'wet','2024 rain threshold 5mm'))
    # Fixed station cohort: preselected using 2023 only. Balanced day x station grid includes true observed zero counts.
    cohort=json.loads((RES/'discovery_station_cohort.json').read_text())
    sd=pd.read_parquet(DATA/'station_daily_2024.parquet')
    sd=sd[sd.start_station_id.isin(cohort)]
    panel=sd.pivot_table(index=['date','start_station_id'],columns='member_casual',values='trips',aggfunc='sum',fill_value=0)
    grid=pd.MultiIndex.from_product([d.index.strftime('%Y-%m-%d'),cohort],names=['date','start_station_id'])
    panel=panel.reindex(grid,fill_value=0)
    station_log=np.log((panel.casual+.5)/(panel.member+.5))
    statmean=station_log.groupby(level='date').mean();statmean.index=pd.to_datetime(statmean.index)
    ds=d.copy();ds['log_ratio']=statmean
    rows.append(estimate(fit(ds),'wet','2024 balanced station panel'))
    # Also compare counts aggregated over the fixed cohort to citywide results.
    fixed=panel.groupby(level='date')[['casual','member']].sum();fixed.index=pd.to_datetime(fixed.index)
    dc=d.copy();dc['log_ratio']=np.log(fixed.casual/fixed.member)
    rows.append(estimate(fit(dc),'wet','2024 fixed-cohort total ratio'))
    pd.DataFrame(rows).to_csv(RES/'validation_effects.csv',index=False)
    # Additional design: one-to-one matched wet / exactly-dry days, without replacement.
    # Same month + same day of week, temperature difference <=3C; optimal maximum-cardinality matching.
    from scipy.optimize import linear_sum_assignment
    pairs=[]
    for (month,dow),sub in d.groupby(['month','dow']):
        a=sub[sub.PRCP>=1];b=sub[sub.PRCP==0]
        if not len(a) or not len(b): continue
        cost=np.abs(a.temp.to_numpy()[:,None]-b.temp.to_numpy()[None,:])
        penalized=np.where(cost<=3,cost,1e6)
        ii,jj=linear_sum_assignment(penalized)
        for i,j in zip(ii,jj):
            if cost[i,j]>3:continue
            wa,db=a.iloc[i],b.iloc[j]
            pairs.append(dict(wet_date=a.index[i].strftime('%Y-%m-%d'),dry_date=b.index[j].strftime('%Y-%m-%d'),
                              month=month,temp_difference=float(wa.temp-db.temp),wind_difference=float(wa.AWND-db.AWND),
                              holiday_difference=float(wa.holiday-db.holiday),snow_difference=float(wa.SNOW-db.SNOW),
                              log_ratio_difference=float(wa.log_ratio-db.log_ratio)))
    pairs=pd.DataFrame(pairs);pairs.to_csv(RES/'matched_pairs.csv',index=False)
    rng=np.random.default_rng(20261112)
    # Resample whole months to retain dependencies among pairs in the same month.
    blocks=[g.log_ratio_difference.to_numpy() for _,g in pairs.groupby('month')]
    boot=np.array([np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]).mean() for _ in range(10000)])
    matched=dict(pairs=len(pairs),months=len(blocks),beta=float(pairs.log_ratio_difference.mean()),
                 ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                 percent=float(100*np.expm1(pairs.log_ratio_difference.mean())),
                 percent_low=float(100*np.expm1(np.quantile(boot,.025))),percent_high=float(100*np.expm1(np.quantile(boot,.975))),
                 mean_abs_temp_difference=float(pairs.temp_difference.abs().mean()),
                 mean_wind_difference=float(pairs.wind_difference.mean()),mean_holiday_difference=float(pairs.holiday_difference.mean()),
                 mean_snow_difference=float(pairs.snow_difference.mean()))
    # Out-of-year prediction: weather-known explanatory prediction, not an operational weather forecast.
    baseline='bs(temp, df=5, degree=3, lower_bound=-40, upper_bound=50) + AWND + np.log1p(SNOW) + np.log1p(SNWD) + C(month) + C(dow) + holiday'
    preds=[]
    for label,form in [('calendar_only','C(month) + C(dow) + holiday'),('weather_without_rain',baseline),('weather_with_rain','wet + '+baseline)]:
        model=smf.ols('log_ratio ~ '+form,data=train).fit()
        p=model.predict(d)
        preds.append(dict(model=label,MAE=float(mean_absolute_error(d.log_ratio,p)),RMSE=float(np.sqrt(mean_squared_error(d.log_ratio,p)))))
        d[f'pred_{label}']=p
    pd.DataFrame(preds).to_csv(RES/'prediction_metrics.csv',index=False)
    d.to_csv(RES/'validation_predictions.csv',index_label='date')
    # Anomaly ranking uses the frozen 2023 full model, never refits on the anomalous day.
    d['residual']=d.log_ratio-d.pred_weather_with_rain
    anomalies=d.loc[d.residual.abs().sort_values(ascending=False).index].head(10)
    anomalies.to_csv(RES/'validation_anomalies.csv',index_label='date')
    qc=pd.read_csv(DATA/'qc_2024.csv')
    summary=dict(year=2024,days=len(d),raw_trips=int(qc.raw_rows.sum()),clean_trips=int(d.total.sum()),wet_days=int(d.wet.sum()),
                 missing_weather=d[['PRCP','TMAX','TMIN','AWND','SNOW','SNWD']].isna().sum().to_dict(),
                 cohort=len(cohort),cohort_present_2024=int(sd.start_station_id.nunique()),
                 effects=rows,matched=matched,prediction=preds,
                 support=bool(rows[0]['beta']<0 and rows[0]['ci_high']<0))
    (RES/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    # Forest plot ratios are percent changes in casual/member, not percentage-point changes in market share.
    plot=pd.DataFrame(rows).loc[lambda x:~x.label.str.contains('demand')].copy()
    labels=['主实验：全市比例','无雨日对照','Midway 天气','宽松时长清洗','仅4—10月','降水阈值5mm','固定站点等权面板','固定站点汇总比例']
    fig,ax=plt.subplots(figsize=(9,5))
    ax.errorbar(plot.percent,range(len(plot)),xerr=[plot.percent-plot.percent_low,plot.percent_high-plot.percent],fmt='o',color='#247b92',capsize=4)
    ax.axvline(0,color='#999',linestyle='--');ax.set(yticks=range(len(plot)),yticklabels=labels,xlabel='雨天临时用户 / 会员行程比的相对变化 (%)',title='2024 年独立验证：调整估计与 95% 置信区间');ax.invert_yaxis()
    fig.savefig(FIG/'07_validation_forest.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(pairs.log_ratio_difference,bins=18,color='#247b92',alpha=.85)
    axes[0].axvline(0,color='#999',linestyle='--');axes[0].set(title=f'匹配日实验：{len(pairs)} 对',xlabel='雨天 − 无雨日：log(临时 / 会员)',ylabel='配对数')
    pm=pd.DataFrame(preds);axes[1].bar(['日历','天气不含降雨','天气含降雨'],pm.RMSE,color=['#aaa','#247b92','#d8763a']);axes[1].set(title='2023 年训练 → 2024 年测试',ylabel='log 比例的 RMSE（越低越好）')
    fig.tight_layout();fig.savefig(FIG/'08_matching_prediction.png');plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['discovery','validation']);args=ap.parse_args()
    discovery() if args.phase=='discovery' else validation()
