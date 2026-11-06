#!/usr/bin/env python3
"""
sigma(P) of the 8 S/m KCl standard from Zachary Diaz's vessel runs, alongside
Cortes's Aug 2025 0.5 m NaCl high-pressure series.

Diaz runs (each fitted from the raw Gamry sweeps, CPE 10-100 kHz):
  20261105  8 S/m KCl standard throughout (Steve, 2026-11-05), 298 K,
            1 -> 200 -> 76 MPa; K from the first three sweeps at ~1 MPa.
  20261014-15  KCl standard sweep, ~296 K (the one in his PNG); K from the
            first three Oct 14 sweeps at ambient, same assumption.
Sweeps with > 5% fit uncertainty in R are kept and drawn hollow. Two kinds:
the fit is unsure but sigma stays on the curve (Oct 14 at 100 MPa, Oct 15
52-100 MPa), and the fit lands off the curve (Oct 14 at 124 MPa, +6.6%; Nov 5
at 152-177 MPa on descent, +2.3%). The raw |Z| at the phase-zero frequency
stays on the curve for all of them (within 0.15%), so the excursions are in
the circuit fit, not the solution; see cortes2026/diaz/README.md.

Left panel: sigma vs P, both datasets on one axis (S/m).
Right panel: sigma / sigma(0.1 MPa) on the same day and isotherm, so the
standard and NaCl pressure dependence can be compared directly.

Usage:  python cortes2026/diaz/plot_standard_sigma_P.py
Writes: cortes2026/diaz/standard_sigma_P_vs_cortes.{pdf,png}
        cortes2026/diaz/diaz_standard_sigma_P.csv
"""
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)
from gamryTools import Solution  # noqa: E402
import cortes_data_processing as cdp  # noqa: E402

HERE = Path(__file__).resolve().parent
F_RANGE_HZ = 1e3 * np.array([10, 100])
SIGMA_STD = 8.0
NOISY_R_UNC = 0.05  # fractional R uncertainty above which a sweep is drawn hollow

# Validated categorical order (blue, orange, aqua, yellow), fixed per entity
COLORS = {'Diaz 2026-11-05 (8 S/m KCl, 25 C)': '#2a78d6',
          'Diaz 2026-10-14/15 (8 S/m KCl, 23 C)': '#eb6834',
          'Cortes 0.5 m NaCl, 20 C': '#1baf7a',
          'Cortes 0.5 m NaCl, 35 C': '#eda100'}


def fit_folder(folders):
    rows = []
    for folder in folders:
        for f in sorted(glob(str(REPO / 'data' / folder / 'Conductivity*' / '*.txt'))):
            s = Solution(cmap_name='viridis')
            s.load_file(f)
            s.fit_circuit(circ_type='CPE', initial_guess=None, print_circuit=False,
                          basin_hopping=False, multiproc=False, f_range_hz=F_RANGE_HZ)
            rows.append({'folder': folder, 'filename': Path(f).name, 'time': s.time,
                         'P_MPa': s.P_MPa, 'T_K': s.T_K,
                         'R_ohm': s.Rcalc_ohm, 'R_unc_ohm': s.Runc_ohm})
    df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    K = SIGMA_STD * df.R_ohm.iloc[:3].mean()
    df['K_cell'] = K
    df['sigma_Sm'] = K / df.R_ohm
    df['noisy'] = df.R_unc_ohm / df.R_ohm > NOISY_R_UNC
    df['leg'] = np.where(df.time.diff().dt.total_seconds().fillna(0).cumsum() >= 0,
                         np.where(df.P_MPa.cummax() > df.P_MPa + 5, 'descent', 'ascent'), '')
    return df


def cortes_nacl():
    d = cdp.load_cortes_data(['20250813Cortes', '20250818Cortes', '20250820Cortes'])
    d = d[(d.comp == 'NaCl') & ((pd.to_numeric(d.w_molal) - 0.5).abs() < 0.01)].copy()
    d['T_C'] = d.T_K - 273.15
    return d


def main():
    nov = fit_folder(['20261105'])
    nov['run'] = 'Diaz 2026-11-05 (8 S/m KCl, 25 C)'
    octo = fit_folder(['20261014', '20261015'])
    octo['run'] = 'Diaz 2026-10-14/15 (8 S/m KCl, 23 C)'
    diaz = pd.concat([nov, octo], ignore_index=True)
    diaz.to_csv(HERE / 'diaz_standard_sigma_P.csv', index=False, float_format='%.6g')
    for run, g in diaz.groupby('run'):
        print(f"{run}: K = {g.K_cell.iloc[0]:.2f}, n = {len(g)}, noisy = {g.noisy.sum()}")

    cn = cortes_nacl()
    series = []
    for run, g in diaz.groupby('run', sort=False):
        series.append((run, g.P_MPa.to_numpy(), g.sigma_Sm.to_numpy(), g.noisy.to_numpy(),
                       g.sigma_Sm.iloc[:3].mean(), g.leg.to_numpy()))
    # Cortes: normalize each day's isotherm to that day's own ambient readings
    cn['day'] = cn.filename.str.extract(r'(2025\d{4})')[0].fillna('20250813')
    for lbl, lo, hi in [('Cortes 0.5 m NaCl, 20 C', 18.5, 20.5), ('Cortes 0.5 m NaCl, 35 C', 34, 36)]:
        g = cn[(cn.T_C > lo) & (cn.T_C < hi)].sort_values('P_MPa')
        ref = g.groupby('day').conductivity_Sm.transform(
            lambda s: s[g.loc[s.index, 'P_MPa'] <= 0.1].mean())
        series.append((lbl, g.P_MPa.to_numpy(), g.conductivity_Sm.to_numpy(),
                       np.zeros(len(g), bool), ref.to_numpy(), np.array(['']*len(g))))

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for lbl, P, S, noisy, ref, leg in series:
        c = COLORS[lbl]
        for ax, y in ((ax1, S), (ax2, S / ref)):
            ok = ~noisy
            desc = leg == 'descent'
            ax.plot(P[ok & ~desc], y[ok & ~desc], 'o', ms=5, color=c, mec='white', mew=0.8,
                    label=lbl if ax is ax1 else None)
            if desc.any():
                ax.plot(P[ok & desc], y[ok & desc], 's', ms=5, color=c, mec='white', mew=0.8)
            if noisy.any():
                ax.plot(P[noisy], y[noisy], 's', ms=6, mfc='none', mec=c, mew=1.4)
    ax1.set_xlabel('Pressure (MPa)')
    ax1.set_ylabel(r'$\sigma$ (S/m)')
    ax1.set_title('Conductivity vs pressure', loc='left')
    ax2.set_xlabel('Pressure (MPa)')
    ax2.set_ylabel(r'$\sigma(P)\,/\,\sigma(0.1\ \mathrm{MPa})$')
    ax2.set_title('Normalized to ambient, same run and isotherm', loc='left')
    ax2.axhline(1, color='#9a9a9a', lw=0.8)
    for ax in (ax1, ax2):
        ax.grid(True, color='#e6e6e6', lw=0.6)
        ax.set_xlim(-10, 720)
    ax1.plot([], [], 's', ms=6, mfc='none', mec='#555555', mew=1.4,
             label=r'R fit uncertainty $>5\%$ (kept)')
    ax1.legend(frameon=False, loc='center right')
    ax2.text(0.98, 0.03, 'circles: ascent   squares: descent', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=8, color='#555555')
    for x in ('pdf', 'png'):
        fig.savefig(HERE / f'standard_sigma_P_vs_cortes.{x}', dpi=200)
    print('Wrote', (HERE / 'standard_sigma_P_vs_cortes.pdf').relative_to(REPO))


if __name__ == '__main__':
    main()
