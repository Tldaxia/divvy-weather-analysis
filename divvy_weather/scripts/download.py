"""Download official sources; preserve hashes and retrieval dates. No redistribution."""
from pathlib import Path
import concurrent.futures as cf
import hashlib, json, time, zipfile
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
RAW.mkdir(parents=True, exist_ok=True)

def fetch(job):
    name, url = job
    p = RAW / name
    if not p.exists():
        for attempt in range(4):
            try:
                part=p.with_suffix(p.suffix+'.part')
                offset=part.stat().st_size if part.exists() else 0
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with requests.get(url, stream=True, timeout=(20, 120),headers=headers) as r:
                    r.raise_for_status()
                    with part.open('ab' if offset and r.status_code==206 else 'wb') as f:
                        for chunk in r.iter_content(1024*1024): f.write(chunk)
                if name.endswith('.zip'):
                    with zipfile.ZipFile(p.with_suffix(p.suffix+'.part')) as z:
                        assert any(n.endswith('.csv') for n in z.namelist())
                p.with_suffix(p.suffix+'.part').rename(p)
                break
            except Exception:
                if attempt == 3: raise
                time.sleep(2**attempt)
    record = dict(file=name, url=url, retrieved_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),
                  bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
    print(f"OK {name}: {record['bytes']:,} bytes",flush=True)
    return record

if __name__ == '__main__':
    jobs = [(f'{y}{m:02d}-divvy-tripdata.zip', f'https://divvy-tripdata.s3.amazonaws.com/{y}{m:02d}-divvy-tripdata.zip') for y in (2023,2024) for m in range(1,13)]
    jobs += [(f'{s}.dly',f'https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{s}.dly') for s in ('USW00094846','USW00014819')]
    jobs += [('ghcnd_readme.txt','https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt'),('station_information.json','https://gbfs.divvybikes.com/gbfs/en/station_information.json')]
    jobs = jobs[24:] + jobs[:24]
    with cf.ThreadPoolExecutor(max_workers=24) as pool:
        records=list(pool.map(fetch,jobs))
    (RAW/'manifest.json').write_text(json.dumps(records,indent=2))
