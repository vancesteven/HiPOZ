#!/usr/bin/env python3
"""
1-bar checks for the large (Bollengier et al. 2019) pressure vessel.

Two uses (Steve, 2026-11-18):

1. Thermal equilibration. With the chiller, a measurement taken within an
   hour of a temperature change of 5 K or more may be made before the
   vessel has equilibrated. Every ambient-pressure step (P < 3 MPa) of the
   KCl standard runs is compared with prior data -- the bottle sigma(T)
   table of the 8 S/m standard (gamryTools.CalStdFit) -- through the cell
   constant it implies, K = sigma_prior(T) * Z. If T in the vessel lags the
   RTD, K departs from that of the equilibrated steps.

   The change time is not logged (the PressTemps logs stop at Nov 6), so it
   is bracketed by the sweeps: the change happened after the last sweep at a
   temperature >= 5 K away and before the first sweep at the new one.
   recent_T_change is 'yes' if even the earliest possible change is < 1 h
   before the step, 'possible' if the bracket straddles 1 h, else 'no'.
   dT_equiv_K converts the K deviation to the temperature error that would
   cause it (d ln sigma / dT of the prior at that T), the size of a lag.

2. Calibration from prior data. After the Nov 16 2026 loading of the 5 wt%
   NaCl sample, a soldered connection to the potentiostat broke, so the cell
   constants in the run's config are not valid. The composition is trusted
   (50.073 ppt = 0.902 mol/kg) and K is set by the run's first 1-bar step
   against prior data: McCleskey (2011) Table 3 NaCl at 0.5 and 1.0 mol/kg,
   interpolated to the run's molality and temperature (molar conductivity
   linear in sqrt(m), then a cubic spline in T). The closing 1-bar step is
   the check.

Usage:  python cortes2026/vessel_1bar_check.py
Writes: cortes2026/diaz/vessel_1bar_check.csv, vessel_1bar_check.{pdf,png}
"""
import os
import re
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from gamryTools import CalStdFit  # noqa: E402
from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012  # noqa: E402

OUT = os.path.join(HERE, 'diaz')
FOLDERS = sorted(os.path.basename(d) for d in glob(os.path.join(REPO, 'data', '202611[0-9][0-9]')))
AMBIENT_MAX_MPA = 3.0
STEP_GAP_S = 300
DT_CHANGE_K = 5.0
EQUIL_H = 1.0
KCL_STD_LABEL = 80000
KCL_STD_MOLAL = 0.699772  # 52.168 g per kg water (gamryTools.ppt_to_molal)

# McCleskey, R. B. (2011), Electrical conductivity of electrolytes found in
# natural waters from (5 to 90) C, J. Chem. Eng. Data 56, 317-327, Table 3
# (mS/cm, 0.1 MPa). Transcribed from conductivity-jpl's
# empirical_conductivity_library.csv (rows 'McCleskey2011 Table 3 single
# electrolyte'); not yet checked against the paper.
MCCLESKEY2011_T_C = np.array([5.0, 10.0, 25.0, 35.0, 45.0, 70.0, 90.0])
MCCLESKEY2011_NACL = {  # mol/kg -> kappa (mS/cm)
    0.5: np.array([29.48, 33.48, 46.52, 56.25, 66.24, 93.50, 116.8]),
    0.9999: np.array([54.44, 61.23, 84.30, 101.9, 119.4, 168.0, 209.4]),
}


def sigma_prior_nacl(m, T_C):
    """NaCl sigma (S/m) at 0.1 MPa from McCleskey (2011), interpolated.

    Molar conductivity kappa/m linear in sqrt(m) between the two tabulated
    molalities (0.5-1.0 mol/kg only), then a cubic spline in T."""
    (m1, k1), (m2, k2) = sorted(MCCLESKEY2011_NACL.items())
    if not m1 <= m <= m2:
        raise ValueError(f'{m} mol/kg is outside the tabulated {m1}-{m2} mol/kg')
    f = (np.sqrt(m) - np.sqrt(m1)) / (np.sqrt(m2) - np.sqrt(m1))
    lam = (k1 / m1) + f * (k2 / m2 - k1 / m1)
    return float(CubicSpline(MCCLESKEY2011_T_C, m * lam)(T_C)) / 10.0


def _mc12(T_C, ions):
    return float(elecCondMcCleskey2012(T_C, {k: {'mols': v} for k, v in ions.items()})['sigma_Sm'][0, 0])


def load_sweeps(folders=FOLDERS):
    """All sweeps of the folders' zAnalysis configs, in time order."""
    frames = []
    for d in folders:
        c = pd.read_csv(os.path.join(REPO, 'data', d, f'zAnalysis{d}.csv'))
        c['t'] = [datetime.strptime(re.search(r'_(\d{8}_\d{6})_', f).group(1), '%Y%m%d_%H%M%S')
                  for f in c.filename]
        c['folder'] = d
        frames.append(c)
    return pd.concat(frames).sort_values('t').reset_index(drop=True)


def ambient_steps(sw):
    """1-bar steps with the bracket on the time since the last >= 5 K change."""
    sw = sw.copy()
    sw['since_min_h'] = np.nan  # earliest possible change -> this sweep
    sw['since_max_h'] = np.nan  # latest possible change -> this sweep
    for i, r in sw.iterrows():
        prev = sw.iloc[:i]
        far = prev[(prev.T_K - r.T_K).abs() >= DT_CHANGE_K]
        if far.empty:
            continue
        after = prev[prev.t > far.t.iloc[-1]]
        first_near = after.t.iloc[0] if len(after) else r.t
        sw.loc[i, 'since_min_h'] = (r.t - first_near).total_seconds() / 3600
        sw.loc[i, 'since_max_h'] = (r.t - far.t.iloc[-1]).total_seconds() / 3600
    amb = sw[sw.P_MPa < AMBIENT_MAX_MPA].copy()
    amb['step'] = ((amb.t.diff().dt.total_seconds() > STEP_GAP_S)
                   | (amb.folder != amb.folder.shift())).cumsum()
    st = amb.groupby('step').agg(
        start=('t', 'first'), folder=('folder', 'first'), comp=('comp', 'first'),
        w_molal=('w_molal', 'first'), T_K=('T_K', 'mean'), Z_Ohm=('Z_Ohm', 'mean'),
        n=('Z_Ohm', 'size'), since_min_h=('since_min_h', 'first'),
        since_max_h=('since_max_h', 'first')).reset_index(drop=True)
    st['T_C'] = st.T_K - 273.15
    st['recent_T_change'] = np.select(
        [st.since_max_h < EQUIL_H, st.since_min_h < EQUIL_H], ['yes', 'possible'], 'no')
    return st


def kcl_check(st):
    """K implied by the bottle table at each KCl 1-bar step, and its
    deviation from the median of the steps not flagged for a recent change."""
    cal = CalStdFit(interpMethod='linear')
    k = st[st.comp == 'KCl'].copy()
    k['sigma_prior_Sm'] = [cal(T, KCL_STD_LABEL) for T in k.T_K]
    k['sigma_MC12_Sm'] = [_mc12(T, {'K_p1': KCL_STD_MOLAL, 'Cl_m1': KCL_STD_MOLAL}) for T in k.T_C]
    k['prior_source'] = '8 S/m bottle table (linear in T)'
    k['K_cell'] = k.sigma_prior_Sm * k.Z_Ohm
    k['K_cell_MC12'] = k.sigma_MC12_Sm * k.Z_Ohm
    K_ref = k.loc[k.recent_T_change == 'no', 'K_cell'].median()
    k['dK_pct'] = 100 * (k.K_cell / K_ref - 1)
    dlns_dT = np.array([np.log(cal(T + 0.5, KCL_STD_LABEL) / cal(T - 0.5, KCL_STD_LABEL)) for T in k.T_K])
    k['dT_equiv_K'] = k.dK_pct / 100 / dlns_dT
    return k, K_ref


def nacl_calibration(st, folder='20261116'):
    """K from the run's first 1-bar step against McCleskey (2011); the
    others are checks against the same prior at their own temperatures."""
    n = st[st.folder == folder].copy()
    m = float(n.w_molal.iloc[0])
    n['sigma_prior_Sm'] = [sigma_prior_nacl(m, T) for T in n.T_C]
    n['sigma_MC12_Sm'] = [_mc12(T, {'Na_p1': m, 'Cl_m1': m}) for T in n.T_C]
    n['prior_source'] = f'McCleskey (2011) NaCl, interpolated to {m:g} mol/kg'
    K = n.sigma_prior_Sm.iloc[0] * n.Z_Ohm.iloc[0]
    n['K_cell'] = K
    n['sigma_Sm'] = K / n.Z_Ohm
    n['dK_pct'] = 100 * (n.sigma_prior_Sm * n.Z_Ohm / K - 1)
    return n, K


def plot(k, K_ref, n, K_nacl, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    sc = ax.scatter(k.start, k.K_cell, c=k.T_C, cmap='coolwarm', vmin=0, vmax=32, s=46,
                    edgecolors='#0b0b0b', linewidths=0.6, zorder=3)
    flag = k[k.recent_T_change != 'no']
    ax.scatter(flag.start, flag.K_cell, s=170, facecolors='none', edgecolors='#0b0b0b',
               linewidths=1.1, zorder=4, label=r'$<$1 h after a $\geq$5 K change (possible)')
    ax.scatter(n.start, n.sigma_prior_Sm * n.Z_Ohm, marker='D', c=n.T_C, cmap='coolwarm',
               vmin=0, vmax=32, s=46, edgecolors='#0b0b0b', linewidths=0.6, zorder=3,
               label=r'NaCl 0.902 m, Nov 16 (first step sets $K$)')
    ax.axhline(K_ref, color='#52514e', lw=0.9, ls='--', zorder=1)
    ax.axhspan(K_ref * 0.99, K_ref * 1.01, color='#e4e3df', zorder=0)
    ax.text(k.start.iloc[0], K_ref * 1.011, rf'KCl median {K_ref:.2f} m$^{{-1}}$ $\pm$1\%',
            fontsize=8, color='#52514e', va='bottom')
    for _, r in k.iterrows():
        ax.annotate(f'{r.T_C:.1f}', (r.start, r.K_cell), xytext=(5, 3),
                    textcoords='offset points', fontsize=7, color='#52514e')
    ax.set_ylabel(r'$K = \sigma_\mathrm{prior}(T)\,Z_\mathrm{1\,bar}$ (m$^{-1}$)')
    ax.set_title('Large vessel, 1-bar steps against prior data (Nov 2026)', fontsize=11)
    fig.colorbar(sc, ax=ax, label=r'$T$ ($^\circ$C)')
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path[:-4] + '.png', dpi=200)
    plt.close(fig)


def main():
    st = ambient_steps(load_sweeps())
    k, K_ref = kcl_check(st)
    n, K_nacl = nacl_calibration(st)
    cols = ['start', 'folder', 'comp', 'w_molal', 'T_C', 'n', 'Z_Ohm', 'since_min_h',
            'since_max_h', 'recent_T_change', 'prior_source', 'sigma_prior_Sm',
            'sigma_MC12_Sm', 'K_cell', 'K_cell_MC12', 'dK_pct', 'dT_equiv_K', 'sigma_Sm']
    out = pd.concat([k, n]).reindex(columns=cols)
    out.round(4).to_csv(os.path.join(OUT, 'vessel_1bar_check.csv'), index=False)
    plot(k, K_ref, n, K_nacl, os.path.join(OUT, 'vessel_1bar_check.pdf'))
    print(out.drop(columns=['prior_source']).round(3).to_string(index=False))
    print(f'KCl reference K {K_ref:.2f} 1/m; Nov 16 NaCl K {K_nacl:.2f} 1/m')


if __name__ == '__main__':
    main()
