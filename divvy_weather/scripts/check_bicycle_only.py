"""Post-hoc schema robustness check, explicitly outside the frozen primary protocol."""
import json
import numpy as np
import pandas as pd
from analyze import ROOT,DATA,RES,load,fit,estimate

def run():
    types=pd.read_parquet(DATA/'daily_type_2024.parquet')
    bike=types[types.rideable_type.isin(['classic_bike','electric_bike','docked_bike'])]
    counts=bike.groupby(['date','member_casual']).trips.sum().unstack(fill_value=0)
    counts.index=pd.to_datetime(counts.index)
    d=load(2024)
    d[['casual','member']]=counts[['casual','member']]
    d['log_ratio']=np.log(d.casual/d.member)
    result=estimate(fit(d),'wet','2024 bicycle-only (post-hoc schema check)')
    result.update(bicycle_trips=int(bike.trips.sum()),excluded_scooter_trips=int(types.loc[types.rideable_type=='electric_scooter','trips'].sum()),
                  provenance='Added after validation revealed electric_scooter in 2024 schema; original frozen primary and its result unchanged.')
    pd.DataFrame([result]).to_csv(RES/'bicycle_only_effect.csv',index=False)
    d.to_csv(RES/'bicycle_only_daily.csv',index_label='date')
    (RES/'bicycle_only_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
