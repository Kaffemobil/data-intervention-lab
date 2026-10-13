"""Reproduce the Census hierarchy intervention shown in the public report.

The Census Port-HS files contain aggregate hierarchy rows beside their children:

- product total:        TOTAL FOR ALL COUNTRIES × TOTAL FOR ALL PORTS
- product-by-port:      TOTAL FOR ALL COUNTRIES × specific port
- product-by-country:   specific country × TOTAL FOR ALL PORTS
- route edge:           specific country × specific port

Treating the parent totals as peer observations mechanically doubles mass. This
audit uses the frozen 2017-2018 vessel-import exposure window and reproduces the
naive 50% top-share pathology, the hierarchy-aware top shares, and exact
accounting identities used to invalidate the first granular run.

Usage
-----
python analysis/census_hierarchy_audit.py /path/to/census_network_data_bundle.zip

Output
------
data/hierarchy_intervention.csv
"""
from __future__ import annotations

import argparse
import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hierarchy_intervention.csv"
HS2 = [29, 30, 38, 39, 40, 62, 64, 73, 74, 76, 82, 83, 84, 85, 87, 90, 95, 96]
YEARS = {2017, 2018}
VALUE = "VES_VAL_MO"
ALL_COUNTRIES = "TOTAL FOR ALL COUNTRIES"
ALL_PORTS = "TOTAL FOR ALL PORTS"


def selected_members(zf: zipfile.ZipFile, hs: int):
    prefix = f"porths_hs{hs:02d}_"
    members = []
    for name in zf.namelist():
        if not name.endswith(".csv.gz") or prefix not in name:
            continue
        # Filenames are porths_hsXX_YYYY-MM_YYYY-MM.csv.gz.
        start_year = int(name.split(prefix, 1)[1][:4])
        if start_year in YEARS:
            members.append(name)
    if len(members) != 2:
        raise RuntimeError(f"expected 2 annual files for HS {hs:02d}, found {len(members)}")
    return sorted(members)


def read_hs(zf: zipfile.ZipFile, hs: int):
    parts = []
    use = ["CTY_CODE", "CTY_NAME", "PORT", "PORT_NAME", VALUE]
    for member in selected_members(zf, hs):
        with zf.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz:
            d = pd.read_csv(
                gz,
                usecols=use,
                dtype={"CTY_CODE": "string", "PORT": "string"},
                low_memory=False,
            )
        d[VALUE] = pd.to_numeric(d[VALUE], errors="coerce").fillna(0.0)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def relative_error(a, b):
    return abs(a - b) / max(abs(a), 1.0)


def audit_hs(d: pd.DataFrame, hs: int, rtol: float):
    all_country = d.CTY_NAME.eq(ALL_COUNTRIES)
    all_port = d.PORT_NAME.eq(ALL_PORTS)

    parent = float(d.loc[all_country & all_port, VALUE].sum())

    by_port = (
        d.loc[all_country & ~all_port]
        .groupby(["PORT", "PORT_NAME"], dropna=False)[VALUE]
        .sum()
    )
    by_country = (
        d.loc[~all_country & all_port]
        .groupby(["CTY_CODE", "CTY_NAME"], dropna=False)[VALUE]
        .sum()
    )
    routes = float(d.loc[~all_country & ~all_port, VALUE].sum())

    port_sum = float(by_port.sum())
    country_sum = float(by_country.sum())

    assert relative_error(parent, port_sum) <= rtol, (hs, "port", parent, port_sum)
    assert relative_error(parent, country_sum) <= rtol, (hs, "country", parent, country_sum)
    assert relative_error(parent, routes) <= rtol, (hs, "route", parent, routes)

    # The invalid parser effectively treated the parent as one more peer.
    naive_port_den = parent + port_sum
    naive_country_den = parent + country_sum
    naive_port_top = max(parent, float(by_port.max())) / naive_port_den
    naive_country_top = max(parent, float(by_country.max())) / naive_country_den

    curated_port_top = float(by_port.max() / parent)
    curated_country_top = float(by_country.max() / parent)

    return {
        "hs": hs,
        "top_port_share_naive": naive_port_top,
        "top_port_share_curated": curated_port_top,
        "top_country_share_naive": naive_country_top,
        "top_country_share_curated": curated_country_top,
        "vessel_value_2017_2018": parent,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="census_network_data_bundle.zip")
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()

    rows = []
    with zipfile.ZipFile(args.bundle) as zf:
        for hs in HS2:
            rows.append(audit_hs(read_hs(zf, hs), hs, args.rtol))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(out.to_string(index=False))
    print("\nmedian top-port share, naive:", f"{out.top_port_share_naive.median():.3f}")
    print("median top-port share, curated:", f"{out.top_port_share_curated.median():.3f}")
    print("median top-country share, naive:", f"{out.top_country_share_naive.median():.3f}")
    print("median top-country share, curated:", f"{out.top_country_share_curated.median():.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
