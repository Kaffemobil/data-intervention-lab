#!/usr/bin/env python3
"""Robust PortWatch monthly downloader.

- No API key required.
- Caches each origin separately, so reruns resume.
- Tries one 2019-2026 grouped query first.
- Falls back to year-by-year, then month-by-month if ArcGIS rejects a query.
- Writes portwatch_monthly_data_bundle.zip for downstream analysis.
"""
from __future__ import annotations
import csv, json, time, urllib.parse, urllib.request, zipfile, hashlib
from pathlib import Path
from datetime import datetime, timezone

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0/query"
ORIGINS = {
    "CHN":"China", "JPN":"Japan", "DEU":"Germany", "KOR":"Korea, South",
    "TWN":"Taiwan", "VNM":"Vietnam", "GBR":"United Kingdom", "ITA":"Italy",
    "IND":"India", "THA":"Thailand", "MEX":"Mexico", "MYS":"Malaysia",
    "IDN":"Indonesia", "FRA":"France",
}
YEARS = list(range(2019, 2027))
OUT = Path("portwatch_monthly_bundle")
CACHE = OUT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

SUM_FIELDS = [
    "portcalls_container", "portcalls_cargo", "portcalls",
    "import_container", "import_cargo", "import",
    "export_container", "export_cargo", "export",
]

def urlopen_json(params, retries=5):
    q = urllib.parse.urlencode(params)
    url = BASE + "?" + q
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                obj = json.load(r)
            if "error" in obj:
                raise RuntimeError(obj["error"])
            return obj
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"PortWatch request failed after {retries} attempts: {last}")

def outstats():
    return json.dumps([
        {"statisticType":"sum","onStatisticField":f,"outStatisticFieldName":f}
        for f in SUM_FIELDS
    ], separators=(",",":"))

def query_grouped(where, group_fields):
    rows = []
    offset = 0
    limit = 1000
    while True:
        params = {
            "where": where,
            "outFields": ",".join(group_fields),
            "groupByFieldsForStatistics": ",".join(group_fields),
            "orderByFields": ",".join(group_fields),
            "outStatistics": outstats(),
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": limit,
        }
        obj = urlopen_json(params)
        feats = obj.get("features", [])
        rows.extend([x.get("attributes", {}) for x in feats])
        if len(feats) < limit:
            break
        offset += limit
    return rows

def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def normalize_origin_rows(rows, iso3, name):
    out = []
    seen = set()
    for r in rows:
        y = int(r["year"])
        m = int(r["month"])
        key = (y, m)
        if key in seen:
            # If ArcGIS ever returns duplicate group labels, sum them deterministically.
            existing = next(x for x in out if int(x["year"]) == y and int(x["month"]) == m)
            for f in SUM_FIELDS:
                existing[f] = (float(existing.get(f) or 0) + float(r.get(f) or 0))
            continue
        seen.add(key)
        row = {"year": y, "month": m, "ISO3": iso3, "country": name}
        for f in SUM_FIELDS:
            row[f] = r.get(f)
        out.append(row)
    return sorted(out, key=lambda x: (int(x["year"]), int(x["month"])))

def download_origin(iso3, name):
    cache_path = CACHE / f"origin_{iso3}.csv"
    fields = ["year","month","ISO3","country"] + SUM_FIELDS

    if cache_path.exists():
        rows = read_csv(cache_path)
        months = {(int(r["year"]), int(r["month"])) for r in rows}
        # Accept cached file if it has at least the expected Jan-2019..current-ish coverage.
        if len(months) >= 80:
            print(f"  resume: {len(rows)} cached country-month rows")
            return rows

    # 1) Fast path: whole period in one request.
    try:
        rows = query_grouped(
            f"ISO3='{iso3}' AND year>=2019 AND year<=2026",
            ["year","month","ISO3"]
        )
        rows = normalize_origin_rows(rows, iso3, name)
        if rows:
            write_csv(cache_path, rows, fields)
            return rows
    except Exception as e:
        print(f"  whole-period query failed: {e}")
        print("  falling back to year-by-year ...", flush=True)

    # 2) Robust path: one year at a time.
    all_rows = []
    for y in YEARS:
        try:
            yr = query_grouped(
                f"ISO3='{iso3}' AND year={y}",
                ["year","month","ISO3"]
            )
            yr = normalize_origin_rows(yr, iso3, name)
            all_rows.extend(yr)
            print(f"    {y}: {len(yr)} months", flush=True)
        except Exception as e:
            print(f"    {y} failed: {e}")
            print(f"    falling back to month-by-month for {y} ...", flush=True)
            # 3) Last-resort path: one month at a time.
            for m in range(1, 13):
                try:
                    mo = query_grouped(
                        f"ISO3='{iso3}' AND year={y} AND month={m}",
                        ["year","month","ISO3"]
                    )
                    mo = normalize_origin_rows(mo, iso3, name)
                    all_rows.extend(mo)
                    if mo:
                        print(f"      {y}-{m:02d}: ok", flush=True)
                except Exception as e2:
                    # Future months in 2026 can legitimately be absent; errors are not silently ignored.
                    print(f"      {y}-{m:02d}: ERROR {e2}", flush=True)

        # Save partial progress after every year.
        merged = {}
        for r in all_rows:
            merged[(int(r["year"]), int(r["month"]))] = r
        partial = [merged[k] for k in sorted(merged)]
        write_csv(cache_path, partial, fields)

    rows = read_csv(cache_path)
    return rows

def main():
    test = urlopen_json({
        "where":"ISO3='JPN' AND year=2019",
        "returnCountOnly":"true",
        "f":"json"
    })
    print("PortWatch/API check passed; JPN 2019 rows:", test.get("count"))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE,
        "origins": ORIGINS,
        "years": YEARS,
        "files": []
    }

    origin_rows = []
    for iso3, name in ORIGINS.items():
        print(f"origin {iso3} {name} ...", flush=True)
        rows = download_origin(iso3, name)
        origin_rows.extend(rows)
        print(f"  {len(rows)} country-month rows", flush=True)

    opath = OUT / "origin_country_monthly.csv"
    ofields = ["year","month","ISO3","country"] + SUM_FIELDS
    write_csv(opath, origin_rows, ofields)
    manifest["files"].append({
        "name": opath.name,
        "rows": len(origin_rows),
        "sha256": sha256(opath)
    })

    # U.S. destination ports: year-by-year already, with per-year cache.
    usa = []
    ufields = ["year","month","portid","portname","ISO3","country"] + SUM_FIELDS
    for y in YEARS:
        cache_path = CACHE / f"usa_ports_{y}.csv"
        if cache_path.exists():
            rows = read_csv(cache_path)
            print(f"USA port-month {y}: resume {len(rows)} cached rows", flush=True)
        else:
            print(f"USA port-month {y} ...", flush=True)
            try:
                rows = query_grouped(
                    f"ISO3='USA' AND year={y}",
                    ["year","month","portid","portname","ISO3","country"]
                )
            except Exception as e:
                print(f"  yearly USA query failed: {e}")
                print("  falling back to month-by-month ...", flush=True)
                rows = []
                for m in range(1, 13):
                    try:
                        mo = query_grouped(
                            f"ISO3='USA' AND year={y} AND month={m}",
                            ["year","month","portid","portname","ISO3","country"]
                        )
                        rows.extend(mo)
                        print(f"    {y}-{m:02d}: {len(mo)} port rows", flush=True)
                    except Exception as e2:
                        print(f"    {y}-{m:02d}: ERROR {e2}", flush=True)
            write_csv(cache_path, rows, ufields)
            print(f"  {len(rows)} port-month rows", flush=True)
        usa.extend(rows)

    upath = OUT / "usa_port_monthly.csv"
    write_csv(upath, usa, ufields)
    manifest["files"].append({
        "name": upath.name,
        "rows": len(usa),
        "sha256": sha256(upath)
    })

    meta_url = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0?f=json"
    try:
        with urllib.request.urlopen(meta_url, timeout=60) as r:
            meta = json.load(r)
    except Exception as e:
        meta = {"metadata_download_error": str(e)}

    mpath = OUT / "arcgis_layer_metadata.json"
    mpath.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest["files"].append({"name": mpath.name, "sha256": sha256(mpath)})

    man = OUT / "manifest.json"
    man.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zpath = Path("portwatch_monthly_data_bundle.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file():
                z.write(p, arcname=f"portwatch_monthly_bundle/{p.name}")

    print("\nDONE:", zpath.resolve())
    print("Bundle ready for downstream analysis.")

if __name__ == "__main__":
    main()
