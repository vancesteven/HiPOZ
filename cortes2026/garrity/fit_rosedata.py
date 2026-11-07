#!/usr/bin/env python3
"""
Provisional analysis of Rose Garrity's 2026-02-19 benchtop run (data/RoseData).

24 Gamry sweeps, three per sample, all at ~296 K and ambient pressure: two KCl
standards (15mS, 2764uS), NaCl at 0.5, 0.75, 1 ('1MNaClRedo'), 1.5 and 2, and
a 'Default' triplet of unknown content. Each sweep is fitted with the same
circuit and frequency band as gamry_HiPOZ.py; the cell constant is
K = sigma_std(T) * R from each standard (bottle-label table, CalStdFit), and
sigma = K / R.

PROVISIONAL. 'M' in her labels is mol/kg (Steve, 2026-11-07: the lab
prepares only molal solutions), so w_molal_nominal is the molality; the 'Default' triplet is unidentified; which standard sets K
is not decided, so both are reported. Not merged into the Cortes data yet.

P is assessed as 0.1 MPa (gauge reads < 10 MPa; see
cortes_data_processing.AMBIENT_P_CUTOFF_MPA).

Usage:  python cortes2026/garrity/fit_rosedata.py
Writes: cortes2026/garrity/rosedata_20260219_fits.csv
"""
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gamryTools import Solution, CalStdFit  # noqa: E402

DATA = REPO / 'data' / 'RoseData' / 'Conductivity'
OUT = Path(__file__).resolve().parent / 'rosedata_20260219_fits.csv'
F_RANGE_HZ = 1e3 * np.array([10, 100])  # as gamry_HiPOZ.py
STANDARDS_USCM = {'15mS': 15000, '2764uS': 2764}
AMBIENT_P_MPA = 0.1


def label_of(path):
    return Path(path).name.split('_')[0]


def nominal_molal(label):
    m = re.match(r'([\d.]+)MNaCl', label)
    return float(m.group(1)) if m else np.nan


def main():
    rows = []
    for f in sorted(glob(str(DATA / '*.txt'))):
        s = Solution(cmap_name='viridis')
        s.load_file(f)
        s.fit_circuit(circ_type='CPE', initial_guess=None, print_circuit=False,
                      basin_hopping=False, multiproc=False, f_range_hz=F_RANGE_HZ)
        lbl = label_of(f)
        rows.append({
            'filename': Path(f).name, 'label': lbl,
            'type': 'standard' if lbl in STANDARDS_USCM else 'measurement',
            'std_uScm': STANDARDS_USCM.get(lbl, np.nan),
            'comp': 'NaCl' if 'NaCl' in lbl else '',
            'w_molal_nominal': nominal_molal(lbl),
            'T_K': s.T_K, 'P_MPa_gauge': s.P_MPa, 'P_MPa': AMBIENT_P_MPA,
            'Z_Ohm': s.Rcalc_ohm, 'Z_unc_Ohm': s.Runc_ohm,
        })
    df = pd.DataFrame(rows)

    cal = CalStdFit(interpMethod='cubic')
    std = df[df.type == 'standard'].copy()
    std['sigma_std_Sm'] = [cal(T, lbl_uScm=int(u)) for T, u in zip(std.T_K, std.std_uScm)]
    std['K_cell'] = std.sigma_std_Sm * std.Z_Ohm
    K = std.groupby('label').K_cell.agg(['mean', 'std', 'count'])
    print('Cell constant by standard (1/m):')
    print(K.round(3).to_string())

    for lbl, k in K['mean'].items():
        df[f'sigma_Sm_K{lbl}'] = k / df.Z_Ohm
    df.loc[std.index, 'K_cell'] = std.K_cell
    df.to_csv(OUT, index=False, float_format='%.6g')

    summ = df.groupby('label', sort=False).agg(
        n=('Z_Ohm', 'size'), T_K=('T_K', 'mean'), Z_Ohm=('Z_Ohm', 'mean'),
        Z_cv_pct=('Z_Ohm', lambda z: 100 * z.std() / z.mean()),
        **{f'sigma_K{l}': (f'sigma_Sm_K{l}', 'mean') for l in K.index})
    print('\nPer sample (means of triplets):')
    print(summ.round(4).to_string())
    print(f'\nWrote {OUT.relative_to(REPO)}')


if __name__ == '__main__':
    main()
