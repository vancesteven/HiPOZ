#!/usr/bin/env python
"""
Before/after comparison of the McCleskey (2012) model against the Cortes
et al. (2026) benchtop conductivities.

Three model variants are scored against every benchtop point with a model:

  A  pre-fix      per-ion ionic strength (the old elecCondMcCleskey2012 bug),
                  mol/L used as mol/kg, mixture specs normalized to sum 1,
                  no speciation -- what the archived figures show
  B  2026-10-08   total ionic strength, mol/L used as mol/kg, old mixture
                  specs, WATEQ4F speciation for single salts only
  C  post-fix     total ionic strength, true molality (molarity_to_molality),
                  mixture specs per first component, WATEQ4F speciation for
                  single salts AND mixtures

plus an independent arbiter: the model vs the IUPAC/OIML Demal KCl primary
conductivity standards, which involve none of the benchtop data.

Speciation uses the repo's Reaktoro path by default (speciation.py). For
machines without Reaktoro, --phreeqc PATH runs the same WATEQ4F speciation
through a USGS PHREEQC executable instead; the two backends were verified to
agree on 2026-10-08.

Outputs (default cortes2026/prepost_comparison/): prepost_table.csv,
prepost_summary.txt, prepost_cortes.{pdf,png}, demal_arbiter.{pdf,png}.

    python cortes2026/compare_prepost_mccleskey.py
    python cortes2026/compare_prepost_mccleskey.py --phreeqc /path/to/phreeqc
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012  # noqa: E402
from cortes_mccleskey import get_mixture_ion_spec  # noqa: E402
import molarity_to_molality as m2m  # noqa: E402

SINGLE = {'NaCl': {'Na_p1': 1, 'Cl_m1': 1}, 'KCl': {'K_p1': 1, 'Cl_m1': 1},
          'Na2SO4': {'Na_p1': 2, 'SO4_m2': 1}}
COMPOUNDS = ['NaCl', 'KCl', 'Na2SO4',
             'Na2SO4:KCl_1:1', 'Na2SO4:KCl_2:1', 'Na2SO4:KCl_1:2',
             'NaCl:MgSO4_1:1', 'NaCl:MgSO4_2:1', 'NaCl:MgSO4_1:2']


def old_mix_spec(compound):
    """The pre-2026-10-17 mixture spec: ratio normalized to sum to 1."""
    r1, r2 = (float(x) for x in compound.split('_')[-1].split(':'))
    f1, f2 = r1 / (r1 + r2), r2 / (r1 + r2)
    if compound.startswith('Na2SO4:KCl'):
        return {'Na_p1': 2 * f1, 'SO4_m2': f1, 'K_p1': f2, 'Cl_m1': f2}
    return {'Na_p1': f1, 'Cl_m1': f1, 'Mg_p2': f2, 'SO4_m2': f2}


class ReaktoroBackend:
    def __init__(self):
        import speciation
        if not speciation.available():
            raise SystemExit('Reaktoro not available; install it '
                             '(conda install -c conda-forge reaktoro) or pass '
                             '--phreeqc PATH')
        self.spec = speciation

    def free(self, ions, T_C):
        inputs = {self.spec.MCCLESKEY_TO_PHREEQC[k]: v for k, v in ions.items()}
        out = self.spec.free_ion_molalities_composition(inputs, T_C + 273.15)
        return out if out is not None else dict(ions)


class PhreeqcBackend:
    ELEMENT = {'Na_p1': 'Na', 'K_p1': 'K', 'Mg_p2': 'Mg', 'Cl_m1': 'Cl',
               'SO4_m2': 'S(6)'}
    SPECIES = {'Na+': 'Na_p1', 'K+': 'K_p1', 'Mg+2': 'Mg_p2', 'Cl-': 'Cl_m1',
               'SO4-2': 'SO4_m2', 'NaSO4-': 'NaSO4_m1', 'KSO4-': 'KSO4_m1',
               'HSO4-': 'HSO4_m1', 'OH-': 'OH_m1', 'H+': 'H_p1'}

    def __init__(self, exe, database=None):
        self.exe = exe
        self.db = database or os.path.join(
            os.path.dirname(os.path.dirname(exe)),
            'share', 'doc', 'phreeqc', 'database', 'wateq4f.dat')
        self.cache = {}

    def free(self, ions, T_C):
        key = (tuple(sorted((k, round(v, 10)) for k, v in ions.items())), T_C)
        if key not in self.cache:
            els = {}
            for k, v in ions.items():
                els[self.ELEMENT[k]] = els.get(self.ELEMENT[k], 0.0) + v
            lines = '\n'.join(f'    {e} {v:.8g}' for e, v in els.items())
            deck = (f'SOLUTION 1\n    temp {T_C}\n    pH 7\n    units mol/kgw\n'
                    f'{lines}\nEND\n')
            with tempfile.TemporaryDirectory() as tmp:
                inp, out = os.path.join(tmp, 'in'), os.path.join(tmp, 'out')
                open(inp, 'w').write(deck)
                subprocess.run([self.exe, inp, out, self.db], check=True,
                               capture_output=True)
                text = open(out).read()
            dist = text.split('Distribution of species')[1].split('Saturation indices')[0]
            free = {}
            for line in dist.splitlines():
                m = re.match(r'^\s{3}(\S+)\s+([\d.e+-]+)\s+[\d.e+-]+', line)
                if m and m.group(1) in self.SPECIES and self.SPECIES[m.group(1)] not in free:
                    free[self.SPECIES[m.group(1)]] = float(m.group(2))
            self.cache[key] = free
        return self.cache[key]


def sigma(ions, T_C, per_ion=False):
    if per_ion:
        return sum(elecCondMcCleskey2012(T_C, {k: {'mols': np.array([v])}})['sigma_Sm'][0][0]
                   for k, v in ions.items())
    return elecCondMcCleskey2012(
        T_C, {k: {'mols': np.array([v])} for k, v in ions.items()})['sigma_Sm'][0][0]


def model(variant, compound, molar, T_C, backend):
    if variant == 'A':
        spec = SINGLE.get(compound) or old_mix_spec(compound)
        return sigma({k: v * molar for k, v in spec.items()}, T_C, per_ion=True)
    if variant == 'B':
        if compound in SINGLE:
            ions = backend.free({k: v * molar for k, v in SINGLE[compound].items()}, T_C)
        else:
            ions = {k: v * molar for k, v in old_mix_spec(compound).items()}
        return sigma(ions, T_C)
    molal = m2m.MOLALITY_TABLE[(compound, round(molar, 4))]
    spec = SINGLE.get(compound) or get_mixture_ion_spec(compound)
    return sigma(backend.free({k: v * molal for k, v in spec.items()}, T_C), T_C)


def score(df, backend):
    molar_col = 'concentration_molar' if 'concentration_molar' in df else 'concentration_molal'
    pts = (df[df.compound.isin(COMPOUNDS)]
           .groupby(['compound', molar_col, 'temperature_C']).conductivity_Sm.mean()
           .reset_index().rename(columns={molar_col: 'molar', 'temperature_C': 'T_C'}))
    rows = []
    for _, r in pts.iterrows():
        rec = dict(compound=r.compound, molar=r.molar,
                   molal=m2m.MOLALITY_TABLE[(r.compound, round(r.molar, 4))],
                   T_C=r.T_C, meas=r.conductivity_Sm)
        for v in 'ABC':
            rec[v] = model(v, r.compound, r.molar, r.T_C, backend)
            rec[f'pct_{v}'] = 100 * (rec[v] - rec['meas']) / rec['meas']
        rows.append(rec)
    return pd.DataFrame(rows)


def summary_text(res):
    lines = [f'{"compound":16s} {"n":>3s}  {"A pre-fix":>13s} {"B 2026-10-08":>13s} '
             f'{"C post-fix":>13s}   RMS % misfit (mean bias %)']
    for c in COMPOUNDS:
        s = res[res.compound == c]
        cells = [f'{np.sqrt((s[f"pct_{v}"] ** 2).mean()):5.1f} ({s[f"pct_{v}"].mean():+5.1f})'
                 for v in 'ABC']
        lines.append(f'{c:16s} {len(s):3d}  ' + ' '.join(f'{x:>13s}' for x in cells))
    return '\n'.join(lines)


def plot_comparison(res, out_dir):
    colors = {5: 'C0', 10: 'C1', 20: 'C2'}
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for ax, c in zip(axes.ravel(), COMPOUNDS):
        s = res[res.compound == c]
        for T in sorted(s.T_C.unique()):
            st = s[s.T_C == T].sort_values('molar')
            col = colors.get(int(T), 'k')
            ax.plot(st.molal, st.meas, 'o', color=col, ms=7)
            ax.plot(st.molar, st.A, ':', color=col, lw=1.5, marker='x', ms=6)
            ax.plot(st.molal, st.C, '-', color=col, lw=2, marker='s', ms=6, mfc='none')
        ax.set_title(c.replace('_', ' '))
        ax.set_xlabel('concentration (data, C: mol/kg; A: as-labeled mol/L)')
        ax.set_ylabel(r'$\sigma$ (S/m)')
        ax.grid(True, ls=':', alpha=0.6)
    axes[0, 0].plot([], [], 'ko', label='benchtop data')
    axes[0, 0].plot([], [], 'k:', marker='x', label='A: pre-fix model')
    axes[0, 0].plot([], [], 'k-', marker='s', mfc='none', label='C: post-fix model')
    axes[0, 0].legend(fontsize=8)
    for T, col in colors.items():
        axes[0, 1].plot([], [], 'o', color=col, label=f'{T} C')
    axes[0, 1].legend(fontsize=8, title='T')
    fig.suptitle('McCleskey model vs Cortes benchtop data: pre-fix (A) vs post-fix (C)\n'
                 'post-fix = total ionic strength + true molality + corrected mixture '
                 'composition + WATEQ4F speciation', fontsize=11)
    fig.tight_layout()
    for xtn in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'prepost_cortes.{xtn}'), dpi=150)
    plt.close(fig)


def plot_demal(out_dir):
    M_KCl = 74.5513
    demal = {'0.1 D': (7.41913, {0: 7.1295, 18: 11.1667, 25: 12.8560}),
             '1 D': (71.1352, {0: 65.144, 18: 97.838, 25: 111.342})}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, (g, kappa) in demal.items():
        m = (g / M_KCl) / ((1000 - g) / 1000)
        Ts = sorted(kappa)
        std = np.array([kappa[T] / 10 for T in Ts])
        pre = np.array([sigma({'K_p1': m, 'Cl_m1': m}, T, per_ion=True) for T in Ts])
        post = np.array([sigma({'K_p1': m, 'Cl_m1': m}, T) for T in Ts])
        ax.plot(Ts, 100 * (pre - std) / std, 'o:', label=f'{name} pre-fix')
        ax.plot(Ts, 100 * (post - std) / std, 's-', label=f'{name} post-fix')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('T (C)')
    ax.set_ylabel('model - standard (%)')
    ax.set_title('Arbiter: McCleskey vs IUPAC/OIML Demal KCl primary standards')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.6)
    fig.tight_layout()
    for xtn in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'demal_arbiter.{xtn}'), dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default=os.path.join(HERE, 'Cortes2026BenchtopData.csv'))
    p.add_argument('--out-dir', default=os.path.join(HERE, 'prepost_comparison'))
    p.add_argument('--phreeqc', metavar='PATH',
                   help='Use a PHREEQC executable for speciation instead of Reaktoro')
    p.add_argument('--database', help='wateq4f.dat path for --phreeqc')
    args = p.parse_args(argv)

    plt.rcParams['text.usetex'] = False
    backend = PhreeqcBackend(args.phreeqc, args.database) if args.phreeqc else ReaktoroBackend()
    os.makedirs(args.out_dir, exist_ok=True)

    res = score(pd.read_csv(args.csv, comment='#'), backend)
    res.round(4).to_csv(os.path.join(args.out_dir, 'prepost_table.csv'), index=False)
    text = summary_text(res)
    open(os.path.join(args.out_dir, 'prepost_summary.txt'), 'w').write(text + '\n')
    print(text)
    plot_comparison(res, args.out_dir)
    plot_demal(args.out_dir)
    print(f'\nWrote outputs to {args.out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
