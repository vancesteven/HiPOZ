#!/usr/bin/env python3
"""
Fill composition and exclusions into the Aug 19-20 2025 Cortes high-pressure
zAnalysis configs from Jesus Cortes's own file labels.

The Gamry files in data/2025081[89]Cortes, data/20250820Cortes are all named
Default_<timestamp>...; Jesus's copies in JesusCortes/Data/08.<dd>.25 carry
descriptive names (JAC_NaCl_0P5m_HP_<P>MPa_<T>C[_flag]). The two are matched
by the timestamp on line 2 of each file. Every matched row gets
comp=NaCl, w_molal=0.5, w_ppt=28.39 (as in the curated Aug 18 config) and a
note naming Jesus's file. Rows are excluded when Jesus flagged them
('inaccurate', 'Takentooearly', 'Donottakefirst...') or when no labelled copy
exists. Calibration is borrowed from Aug 18 separately:

    python gamry_HiPOZ.py --headless --dates 20250819Cortes 20250820Cortes \
        --cal-map 20250819Cortes=20250818Cortes 20250820Cortes=20250818Cortes

Usage:  python cortes2026/curate_aug2025_hp.py [--root .] [--dry-run]
"""
import argparse
import os
import re

import pandas as pd

FOLDERS = {'20250819Cortes': '08.19.25', '20250820Cortes': '08.20.25'}
COMP, W_MOLAL, W_PPT = 'NaCl', 0.5, 28.39
FLAG = re.compile(r'inn?acc?urate|takentooearly|donottakefirst', re.I)


def timestamp(path):
    with open(path, errors='ignore') as f:
        f.readline()
        return f.readline().strip()


def jesus_labels(root, jesus_dir):
    labels = {}
    d = os.path.join(root, 'JesusCortes', 'Data', jesus_dir)
    for name in os.listdir(d):
        if name.endswith('.txt'):
            labels[timestamp(os.path.join(d, name))] = name
    return labels


def curate(root, folder, jesus_dir):
    data_dir = os.path.join(root, 'data', folder)
    cfg = os.path.join(data_dir, f'zAnalysis{folder}.csv')
    df = pd.read_csv(cfg)
    labels = jesus_labels(root, jesus_dir)
    for col in ('comp', 'notes', 'exclude'):
        df[col] = df[col].astype(object)
    for i, row in df.iterrows():
        if row.get('type') == 'standard':
            continue
        path = os.path.join(data_dir, 'ConductivityData_Default', row['filename'])
        label = labels.get(timestamp(path)) if os.path.exists(path) else None
        if label is None:
            df.loc[i, ['exclude', 'notes']] = [True, 'no labelled copy in JesusCortes/Data']
            continue
        df.loc[i, ['comp', 'w_ppt', 'w_molal']] = [COMP, W_PPT, W_MOLAL]
        flagged = bool(FLAG.search(label))
        df.loc[i, 'exclude'] = flagged
        df.loc[i, 'notes'] = (f'Jesus: {label}' + (' (flagged by Jesus)' if flagged else ''))
    return cfg, df


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--root', default=os.path.join(os.path.dirname(__file__), '..'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    for folder, jesus_dir in FOLDERS.items():
        cfg, df = curate(a.root, folder, jesus_dir)
        meas = df[df['type'] != 'standard']
        print(f'{folder}: {len(meas)} measurements, '
              f'{int(meas["exclude"].eq(True).sum())} excluded')
        if not a.dry_run:
            df.to_csv(cfg, index=False)


if __name__ == '__main__':
    main()
