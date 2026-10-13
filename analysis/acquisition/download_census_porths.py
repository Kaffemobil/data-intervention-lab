#!/usr/bin/env python3
"""Download the official U.S. Census monthly Port x HS x country import data
needed for the granular logistics-price forecasting study.

Security: the API key is read from CENSUS_API_KEY and is NEVER written to disk.
Dependencies: Python standard library only.

Default scope: the 18 pre-specified HS2 manufactured-goods chapters from 2010-01
through 2026-06. The script first tries one API request per HS2/year and
falls back to quarter/month chunks if a response is too large.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.census.gov/data/timeseries/intltrade/imports/porths"
HS2_DEFAULT = [29,30,38,39,40,62,64,73,74,76,82,83,84,85,87,90,95,96]
FIELDS = [
    "I_COMMODITY","CTY_CODE","CTY_NAME","PORT","PORT_NAME","YEAR","MONTH",
    "GEN_VAL_MO","AIR_VAL_MO","AIR_WGT_MO","VES_VAL_MO","VES_WGT_MO",
    "CNT_VAL_MO","CNT_WGT_MO","LAST_UPDATE"
]
UA = "logistics-price-research/1.0 (public scientific data download)"

@dataclass(frozen=True)
class Month:
    y: int
    m: int
    def __str__(self): return f"{self.y:04d}-{self.m:02d}"

def parse_month(s: str) -> Month:
    try:
        y, m = map(int, s.split("-"))
        if not 1 <= m <= 12: raise ValueError
        return Month(y,m)
    except Exception:
        raise argparse.ArgumentTypeError("month must be YYYY-MM")

def month_ord(x: Month) -> int: return x.y*12 + x.m - 1

def from_ord(n: int) -> Month: return Month(n//12, n%12 + 1)

def month_range(a: Month,b: Month):
    return [from_ord(i) for i in range(month_ord(a), month_ord(b)+1)]

def chunks_year_first(start: Month, end: Month):
    """Return year-sized chunks respecting partial first/last years."""
    out=[]
    cur=start
    while month_ord(cur) <= month_ord(end):
        chunk_end = Month(cur.y, 12)
        if month_ord(chunk_end)>month_ord(end): chunk_end=end
        out.append((cur,chunk_end))
        cur=from_ord(month_ord(chunk_end)+1)
    return out

def split_chunk(a: Month,b: Month):
    months=month_range(a,b)
    if len(months)<=1: return []
    # Annual -> quarterly-ish chunks; smaller chunks -> individual months.
    if len(months)>3:
        return [(months[i], months[min(i+2,len(months)-1)]) for i in range(0,len(months),3)]
    return [(m,m) for m in months]

def safe_url(params: dict) -> str:
    red=dict(params)
    if "key" in red: red["key"]="<REDACTED>"
    return API + "?" + urllib.parse.urlencode(red)

def actual_url(params: dict) -> str:
    return API + "?" + urllib.parse.urlencode(params)

def valid_csv(blob: bytes) -> tuple[bool,str,int]:
    try:
        txt=blob.decode("utf-8-sig",errors="replace")
        low=txt[:1000].lower()
        if "error" in low and "i_commodity" not in low:
            return False, txt[:500].replace("\n"," "), 0
        r=csv.reader(io.StringIO(txt))
        header=next(r,None)
        if not header or "I_COMMODITY" not in header or "PORT" not in header:
            return False, txt[:500].replace("\n"," "), 0
        n=sum(1 for _ in r)
        return True,"",n
    except Exception as e:
        return False,f"CSV parse error: {e}",0

def fetch(params: dict, attempts=5, timeout=180) -> tuple[bytes,int]:
    url=actual_url(params)
    last=None
    for k in range(attempts):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept":"text/csv,*/*"})
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                blob=resp.read()
            ok,msg,n=valid_csv(blob)
            if ok: return blob,n
            raise RuntimeError(msg)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last=e
            if isinstance(e, urllib.error.HTTPError) and e.code in (400,404,413,414):
                break
            time.sleep(min(30, 2**k))
    raise RuntimeError(str(last))

def write_gz(path: Path, blob: bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wb",compresslevel=9) as f: f.write(blob)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start",type=parse_month,default=parse_month("2010-01"))
    ap.add_argument("--end",type=parse_month,default=parse_month("2026-06"))
    ap.add_argument("--out",type=Path,default=Path("census_network_data"))
    ap.add_argument("--hs2",nargs="*",type=int,default=HS2_DEFAULT)
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    if month_ord(args.end)<month_ord(args.start): ap.error("--end precedes --start")

    key=os.environ.get("CENSUS_API_KEY","").strip()
    if not key:
        print("ERROR: CENSUS_API_KEY is not set. Do not paste the key into this script or chat.",file=sys.stderr)
        print("Set it as an environment variable, then rerun. See README.md.",file=sys.stderr)
        return 2

    root=args.out.resolve(); raw=root/"raw"; raw.mkdir(parents=True,exist_ok=True)
    manifest_path=root/"manifest.jsonl"
    log=[]
    if manifest_path.exists() and not args.force:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try: log.append(json.loads(line))
            except Exception: pass
    completed={x.get("file") for x in log if x.get("status")=="ok"}

    # Key sanity check on one small official query.
    sanity={"get":"I_COMMODITY,PORT,CTY_CODE,YEAR,MONTH,GEN_VAL_MO",
            "time":str(args.start),"COMM_LVL":"HS2","I_COMMODITY":f"{args.hs2[0]:02d}",
            "SUMMARY_LVL":"DET","key":key,"outputFormat":"csv"}
    try:
        _, n=fetch(sanity,attempts=2,timeout=60)
        print(f"API/key check passed ({n:,} rows for {args.hs2[0]:02d}, {args.start}).")
    except Exception as e:
        print(f"ERROR: Census API/key check failed: {e}",file=sys.stderr)
        return 3

    def download_segment(hs:int,a:Month,b:Month):
        fname=f"porths_hs{hs:02d}_{a}_{b}.csv.gz"
        if fname in completed and (raw/fname).exists() and not args.force:
            print("skip",fname)
            return
        params={"get":",".join(FIELDS),
                "time": str(a) if a==b else f"from {a} to {b}",
                "COMM_LVL":"HS2","I_COMMODITY":f"{hs:02d}","SUMMARY_LVL":"DET",
                "key":key,"outputFormat":"csv"}
        try:
            blob,n=fetch(params)
            sha=hashlib.sha256(blob).hexdigest(); write_gz(raw/fname,blob)
            rec={"status":"ok","file":fname,"hs2":f"{hs:02d}","start":str(a),"end":str(b),
                 "rows":n,"bytes_csv":len(blob),"sha256_csv":sha,"query":safe_url(params),
                 "retrieved_at_utc":datetime.now(timezone.utc).isoformat()}
            with manifest_path.open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
            completed.add(fname)
            print(f"ok   {fname}: {n:,} rows")
        except Exception as e:
            subs=split_chunk(a,b)
            if not subs:
                rec={"status":"error","file":fname,"hs2":f"{hs:02d}","start":str(a),"end":str(b),
                     "error":str(e),"query":safe_url(params),"retrieved_at_utc":datetime.now(timezone.utc).isoformat()}
                with manifest_path.open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
                raise RuntimeError(f"failed irreducible query {hs:02d} {a}: {e}")
            print(f"split {hs:02d} {a}..{b} after API response: {e}")
            for sa,sb in subs: download_segment(hs,sa,sb)

    try:
        for hs in args.hs2:
            if not 1<=hs<=97: raise ValueError(f"invalid HS2 {hs}")
            for a,b in chunks_year_first(args.start,args.end):
                download_segment(hs,a,b)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is resumable; rerun the same command.",file=sys.stderr); return 130
    except Exception as e:
        print(f"\nERROR: {e}\nProgress is saved; rerun the same command to resume.",file=sys.stderr); return 4

    metadata={
        "dataset":"U.S. Census Time Series International Trade: Monthly U.S. Imports by Port and HS Code",
        "endpoint":API,"scope":{"hs2":[f"{x:02d}" for x in args.hs2],"start":str(args.start),"end":str(args.end)},
        "fields":FIELDS,"query_filters":{"COMM_LVL":"HS2","SUMMARY_LVL":"DET"},
        "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
        "security":"Census API key was read from CENSUS_API_KEY and is not stored in this bundle.",
        "scientific_note":"Current/revised Census history. Retrospective forecasting must impose publication lags and be labelled pseudo-real-time unless historical vintages are separately archived."
    }
    (root/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    # Include this downloader itself for provenance.
    try: (root/"downloader_used.py").write_bytes(Path(__file__).read_bytes())
    except Exception: pass

    bundle=root.parent/(root.name+"_bundle.zip")
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(root.parent))
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    print("\nDONE")
    print("Bundle:",bundle)
    print("SHA256:",digest)
    print("Bundle contains public trade data and no API key.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
