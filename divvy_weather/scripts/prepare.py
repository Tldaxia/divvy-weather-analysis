"""Chunked trip QC and aggregation, and NOAA fixed-width daily parsing."""
from pathlib import Path
import argparse, calendar, json, zipfile, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
OUT.mkdir(exist_ok=True)

def weather():
    for station in ['USW00094846','USW00014819']:
        records=[]
        for line in (ROOT/'data/raw'/f'{station}.dly').read_text().splitlines():
            year,month,element=int(line[11:15]),int(line[15:17]),line[17:21]
            if year not in (2023,2024) or element not in ['PRCP','SNOW','SNWD','AWND','TMAX','TMIN']: continue
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                cell=line[21+(day-1)*8:29+(day-1)*8]
                value=int(cell[:5]); mflag,qflag,sflag=cell[5:8]
                valid=value != -9999 and qflag == ' '
                scale=10 if element in ['PRCP','AWND','TMAX','TMIN'] else 1
                records.append(dict(date=f'{year}-{month:02d}-{day:02d}',element=element,value=value/scale if valid else np.nan,
                                    mflag=mflag.strip(),qflag=qflag.strip(),sflag=sflag.strip()))
        raw=pd.DataFrame(records)
        raw.to_csv(OUT/f'weather_flags_{station}.csv',index=False)
        w=raw.pivot(index='date',columns='element',values='value').reindex(index=pd.date_range('2023-01-01','2024-12-31').strftime('%Y-%m-%d'),columns=['PRCP','SNOW','SNWD','AWND','TMAX','TMIN'])
        w.index.name='date'
        w.to_csv(OUT/f'weather_{station}.csv')
        print(station,w.isna().sum().to_dict(),flush=True)

def trips(year):
    daily=[]; hourly=[]; stations=[]; inventory=[]; sensitivity=[]; audits=[]; types=[]
    # Duplicates are checked across every monthly file in each year, not merely each chunk.
    seen=set()
    for month in range(1,13):
        name=f'{year}{month:02d}-divvy-tripdata.zip'
        audit=dict(year=year,month=month,raw_rows=0,duplicate_id=0,invalid_time=0,invalid_duration=0,unknown_member=0,
                   wrong_month=0,missing_start_station_raw=0,missing_start_station_clean=0,clean_rows=0)
        while not (ROOT/'data/raw'/name).exists():
            print('Waiting for',name,flush=True);time.sleep(20)
        with zipfile.ZipFile(ROOT/'data/raw'/name) as z:
            csvs=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
            for csv in csvs:
                with z.open(csv) as f:
                    for d in pd.read_csv(f,chunksize=200000,dtype={'ride_id':'str','start_station_id':'str','end_station_id':'str'}):
                        audit['raw_rows']+=len(d)
                        audit['missing_start_station_raw']+=int(d.start_station_id.isna().sum())
                        # Avoid rebuilding a multi-million-ID index for every chunk.
                        dup=d.ride_id.map(seen.__contains__)|d.ride_id.duplicated()
                        seen.update(d.ride_id.dropna())
                        audit['duplicate_id']+=int(dup.sum())
                        d=d.loc[~dup].copy()
                        start=pd.to_datetime(d.started_at,format='mixed',errors='coerce')
                        end=pd.to_datetime(d.ended_at,format='mixed',errors='coerce')
                        bad=start.isna()|end.isna()|d.ride_id.isna()
                        audit['invalid_time']+=int(bad.sum())
                        duration=(end-start).dt.total_seconds()
                        duration_bad=(duration<60)|(duration>86400)
                        audit['invalid_duration']+=int((duration_bad&~bad).sum())
                        member_bad=~d.member_casual.isin(['casual','member'])
                        audit['unknown_member']+=int((member_bad&~bad&~duration_bad).sum())
                        wrong=(start.dt.year!=year)|(start.dt.month!=month)
                        audit['wrong_month']+=int((wrong&~bad&~duration_bad&~member_bad).sum())
                        valid=~bad&~member_bad&~wrong
                        d['date']=start.dt.strftime('%Y-%m-%d');d['hour']=start.dt.hour
                        # Broad-duration sensitivity keeps >0 through 7 days; no station requirement.
                        broad=d.loc[valid&(duration>0)&(duration<=604800)]
                        sensitivity.append(broad.groupby(['date','member_casual']).size().rename('trips').reset_index())
                        d=d.loc[valid&~duration_bad].copy()
                        audit['clean_rows']+=len(d)
                        audit['missing_start_station_clean']+=int(d.start_station_id.isna().sum())
                        daily.append(d.groupby(['date','member_casual']).size().rename('trips').reset_index())
                        types.append(d.groupby(['date','rideable_type','member_casual']).size().rename('trips').reset_index())
                        hourly.append(d.groupby(['date','hour','member_casual']).size().rename('trips').reset_index())
                        named=d.dropna(subset=['start_station_id'])
                        stations.append(named.groupby(['date','start_station_id','member_casual']).size().rename('trips').reset_index())
                        inventory.append(named.groupby(['start_station_id','start_station_name'],dropna=False).agg(trips=('ride_id','size'),lat=('start_lat','median'),lon=('start_lng','median')).reset_index())
        audits.append(audit)
        print(name,audit,flush=True)
    for label,parts,keys in [('daily',daily,['date','member_casual']),('hourly',hourly,['date','hour','member_casual']),
                             ('station_daily',stations,['date','start_station_id','member_casual']),('daily_broad',sensitivity,['date','member_casual']),
                             ('daily_type',types,['date','rideable_type','member_casual'])]:
        result=pd.concat(parts).groupby(keys,observed=True)['trips'].sum().reset_index()
        result.to_parquet(OUT/f'{label}_{year}.parquet',index=False)
    pd.DataFrame(audits).to_csv(OUT/f'qc_{year}.csv',index=False)
    inv=pd.concat(inventory)
    inv.groupby(['start_station_id','start_station_name'],dropna=False).agg(trips=('trips','sum'),lat=('lat','median'),lon=('lon','median')).reset_index().to_csv(OUT/f'station_inventory_{year}.csv',index=False)
    assert sum(x['clean_rows'] for x in audits)==sum(x.trips.sum() for x in daily)
    for x in audits:
        assert x['raw_rows']==sum(x[k] for k in ['duplicate_id','invalid_time','invalid_duration','unknown_member','wrong_month','clean_rows'])
    print('PASS aggregation conservation',year,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--year',type=int);ap.add_argument('--weather',action='store_true');args=ap.parse_args()
    if args.weather: weather()
    if args.year: trips(args.year)
