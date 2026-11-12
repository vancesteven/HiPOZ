#!/usr/bin/env python3
"""
Refit Diaz's 8 S/m KCl standard sweeps to test whether the sigma(P)
excursions are circuit-fit jumps (cortes2026/diaz/README.md).

Same circuit (R_0-p(R_1-CPE_1,C_1)) and band (10-100 kHz) as
gamryTools.Solution.fit_circuit, three ways per sweep:
  default  the production initial guess [Re Z(f_max), 0.02, 1e-5, 0.9, 1e-6]
  chained  start from the previous sweep's chained-fit parameters
  best     lowest residual among default, chained and a fixed set of starts
R from each is compared with the model-free |Z| at phase zero.

Usage:  python cortes2026/diaz/refit_chained.py
Writes: cortes2026/diaz/refit_chained.csv, refit_chained.{pdf,png}
"""
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from gamryTools import Solution  # noqa: E402

HERE = Path(__file__).resolve().parent
CIRC = 'R_0-p(R_1-CPE_1,C_1)'
F_RANGE_HZ = (1e4, 1e5)
RUNS = {'Oct 14-15': ['20261014', '20261015'], 'Nov 5': ['20261105']}
EXTRA_STARTS = [[None, 1.0, 1e-5, 0.9, 1e-6], [None, 10.0, 1e-5, 0.8, 1e-6],
                [None, 0.02, 1e-4, 0.7, 1e-6], [None, 5.0, 1e-6, 0.95, 1e-7]]


def load(f):
    s = Solution(cmap_name='viridis')
    s.load_file(f)
    a = np.loadtxt(f, skiprows=10, usecols=(1, 2, 3))
    a = a[np.argsort(a[:, 0])]
    m = (a[:, 0] > 2e4) & (a[:, 0] < 5e5)
    ph, z = a[m, 2], a[m, 1]
    i = np.where(np.diff(np.sign(ph)) != 0)[0][0]
    R0 = z[i] - ph[i] / (ph[i + 1] - ph[i]) * (z[i + 1] - z[i])
    band = (s.f_Hz > F_RANGE_HZ[0]) & (s.f_Hz <= F_RANGE_HZ[1])
    return s, s.f_Hz[band], s.Z_ohm[band], np.real(s.Z_ohm[0]), R0


def fit(f, Z, guess):
    c = CustomCircuit(CIRC, initial_guess=list(guess))
    try:
        c.fit(f, Z)
    except Exception:
        return None
    res = np.sqrt(np.mean(np.abs(c.predict(f) - Z) ** 2))
    return {'p': c.parameters_, 'R': c.parameters_[0], 'unc': c.conf_[0], 'rms': res}


def main():
    rows = []
    for run, folders in RUNS.items():
        files = [f for d in folders for f in glob(str(REPO / 'data' / d / 'Conductivity*' / '*.txt'))]
        loaded = sorted((load(f) + (f,) for f in files), key=lambda t: t[0].time)
        prev = None
        for s, f, Z, ReZ0, Rph, path in loaded:
            default = fit(f, Z, [ReZ0, 0.02, 1e-5, 0.9, 1e-6])
            chained = fit(f, Z, prev) if prev is not None else default
            prev = chained['p'] if chained else prev
            cands = [x for x in [default, chained] + [fit(f, Z, [ReZ0] + g[1:]) for g in EXTRA_STARTS] if x]
            best = min(cands, key=lambda x: x['rms'])
            rows.append({'run': run, 'filename': Path(path).name, 'time': s.time,
                         'P_MPa': s.P_MPa, 'T_K': s.T_K, 'R_phase0': Rph,
                         **{f'{k}_{q}': v[q] for k, v in (('default', default), ('chained', chained), ('best', best))
                            for q in ('R', 'unc', 'rms')}})
    df = pd.DataFrame(rows)
    for k in ('default', 'chained', 'best', 'phase0'):
        col = 'R_phase0' if k == 'phase0' else f'{k}_R'
        df[f'sigma_{k}'] = df.groupby('run')[col].transform(lambda R: 8.0 * R.iloc[:3].mean() / R)
    df.to_csv(HERE / 'refit_chained.csv', index=False, float_format='%.6g')

    for run, g in df.groupby('run', sort=False):
        jump = (g.default_unc / g.default_R > 0.05)
        print(f'== {run}: {jump.sum()} sweeps with default R unc > 5%')
        for k in ('default', 'chained', 'best'):
            dev = 100 * (g[f'sigma_{k}'] / g.sigma_phase0 - 1)
            hi = (g[f'{k}_unc'] / g[f'{k}_R'] > 0.05).sum()
            print(f'  {k:8s} max |sigma - sigma_phase0| = {dev.abs().max():.2f}%  '
                  f'(on flagged: {dev[jump].abs().max():.2f}%), sweeps with unc > 5%: {hi}, '
                  f'median rms {g[f"{k}_rms"].median():.4f} ohm')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=False)
    style = {'default': ('#2a78d6', 's', 'production fit'), 'chained': ('#eb6834', 'o', 'chained initial guess'),
             'phase0': ('#555555', '+', r'$|Z|$ at phase zero (no model)')}
    for ax, (run, g) in zip(axs, df.groupby('run', sort=False)):
        x = np.arange(len(g))
        for k, (c, mk, lbl) in style.items():
            ax.plot(x, g[f'sigma_{k}'], mk, color=c, ms=6 if mk != '+' else 8,
                    mfc='none' if k == 'default' else c, mew=1.3, label=lbl)
        ax.set_title(f'{run}: 8 S/m KCl standard', loc='left')
        ax.set_xlabel('Sweep number (time order)')
        ax.set_ylabel(r'$\sigma$ (S/m)')
        ax.grid(True, color='#e6e6e6', lw=0.6)
        P = g.P_MPa.to_numpy()
        for i in range(0, len(g), 3):
            ax.annotate(f'{P[i]:.0f}', (x[i], ax.get_ylim()[0]), fontsize=7, color='#777777',
                        xytext=(0, 3), textcoords='offset points')
    axs[0].legend(frameon=False, fontsize=8, loc='upper left')
    axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc='upper left')
    fig.text(0.5, -0.01, 'Grey numbers: pressure (MPa) at each triplet', ha='center',
             va='top', fontsize=8, color='#777777')
    for x in ('pdf', 'png'):
        fig.savefig(HERE / f'refit_chained.{x}', dpi=200, bbox_inches='tight')
    print('Wrote cortes2026/diaz/refit_chained.{csv,pdf,png}')


if __name__ == '__main__':
    main()
