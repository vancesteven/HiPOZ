#!/usr/bin/env python3
"""
Mark the excluded rows in the Sep-Oct 2026 zAnalysis configs imported from
Zachary Diaz's workbooks (import_zanalysis_workbook.py). Run after importing.
The reasons are in docs/DIAZ_2026_ANALYSIS.md.

Each excluded row gets exclude='x' and its reason appended to the notes; the
CSV is then harmonized to JSON. Running it again changes nothing.

Usage:  python curate_diaz_2026.py [--root .] [--dry-run]
"""
import argparse
import os

import pandas as pd

# folder -> list of (filename regex, reason)
EXCLUSIONS = {
    '20260923': [
        (r'_20260923_17(?:4008|4102|4155)_',
         'excluded: end of Trial 2, Z 45-48 ohm vs 23.6 ohm minutes earlier '
         'at the same 25 C setpoint (sigma ~50% low)'),
    ],
    '20261009': [
        (r'_20261009_',
         'excluded: sigma ~20% below 5 wt% NaCl literature; no standard row, '
         'implied K = 100.2 vs ~119.6 in September (cell constant unresolved)'),
    ],
}


def curate(root, folder, rules):
    cfg = os.path.join(root, 'data', folder, f'zAnalysis{folder}.csv')
    df = pd.read_csv(cfg, dtype={'exclude': object, 'notes': object})
    hits = 0
    for pattern, reason in rules:
        mask = df['filename'].str.contains(pattern, regex=True)
        if not mask.any():
            raise ValueError(f'{cfg}: no rows match {pattern}')
        for i in df.index[mask]:
            notes = '' if pd.isna(df.at[i, 'notes']) else str(df.at[i, 'notes'])
            if reason not in notes:
                df.at[i, 'notes'] = f'{notes}; {reason}' if notes else reason
            df.at[i, 'exclude'] = 'x'
        hits += int(mask.sum())
    return cfg, df, hits


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    for folder, rules in EXCLUSIONS.items():
        cfg, df, hits = curate(a.root, folder, rules)
        print(f'{folder}: {hits} of {len(df)} rows excluded')
        if a.dry_run:
            continue
        df.to_csv(cfg, index=False)
        from pathlib import Path
        from harmonize_config import harmonize_config
        harmonize_config(Path(cfg), force=True, verbose=False)


if __name__ == '__main__':
    main()
