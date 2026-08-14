#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate WATEQ4F speciation against benchtop conductivity measurements.

Compares, per compound and temperature:

    measured   -- benchtop conductivity
    MC12       -- McCleskey (2012) evaluated on TOTAL molality (what this repo
                  did before speciation was available; assumes full dissociation)
    MC12+spec  -- McCleskey (2012) evaluated on FREE charged species from a
                  WATEQ4F speciation, which is what MC12 actually specifies

Run this and read the output *before* changing any figure.

    python validate_speciation.py                # Mahboub dataset
    python validate_speciation.py --dataset both

Requires Reaktoro for the speciated column (conda-forge only); the MC12 column
works without it.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012
from study_plots import ION_SPECS
import speciation as spec

MAHBOUB_CSV = 'mahboub2026/Mahboub2026BenchtopData.csv'
CORTES_CSV = 'cortes2026/Cortes2026BenchtopData.csv'

# Compounds that WATEQ4F returns fully dissociated, so speciation must be a
# no-op. These are the calibration-standard salts: any change here is a bug.
NO_OP_COMPOUNDS = ('KCl', 'NaCl', 'NH4Cl')


def sigma_total(compound, m, T_K):
    """MC12 on total molality (full dissociation), as the repo did before."""
    ion_spec = ION_SPECS[compound]
    ions = {ion: {'mols': np.array([m * mult])}
            for ion, mult in ion_spec.items()}
    out = elecCondMcCleskey2012(float(T_K - 273.15), ions)['sigma_Sm']
    return float(np.atleast_1d(out).ravel()[0])


def sigma_speciated(compound, m, T_K):
    """MC12 on free charged species from a WATEQ4F speciation."""
    ions = spec.speciated_ions_dict(compound, [m], T_K)
    if not ions:
        return float('nan')
    out = elecCondMcCleskey2012(float(T_K - 273.15), ions)['sigma_Sm']
    return float(np.atleast_1d(out).ravel()[0])


def load(path, compound_col='compound'):
    try:
        df = pd.read_csv(path, comment='#')
    except FileNotFoundError:
        print(f'  (skipped: {path} not found)')
        return None
    needed = {compound_col, 'concentration_molal', 'temperature_K',
              'conductivity_Sm'}
    if not needed.issubset(df.columns):
        print(f'  (skipped: {path} lacks columns {sorted(needed - set(df.columns))})')
        return None
    return df.groupby([compound_col, 'concentration_molal', 'temperature_K'],
                      as_index=False)['conductivity_Sm'].mean()


def report(df, temp_K=None, use_speciation=True):
    """Print the comparison table, grouped by compound."""
    rows = []
    for compound in sorted(df['compound'].unique()):
        if compound not in ION_SPECS:
            continue
        sub = df[df['compound'] == compound]
        if temp_K is not None:
            sub = sub[np.isclose(sub['temperature_K'], temp_K, atol=1.5)]
        if not len(sub):
            continue

        speciable = use_speciation and compound in spec.SALT_RECIPES
        print(f'\n{compound}'
              f'{"" if speciable else "   (no speciation recipe -- MC12 only)"}')
        print(f'  {"molal":>8} {"T(K)":>7} {"meas":>9} {"MC12":>9} '
              f'{"MC12+spec":>10} {"err_MC12":>9} {"err_spec":>9} {"free":>7}')

        for _, r in sub.sort_values(['concentration_molal',
                                     'temperature_K']).iterrows():
            m, T_K, meas = (r['concentration_molal'], r['temperature_K'],
                            r['conductivity_Sm'])
            if meas <= 0:
                # e.g. 0.0249 molal MgSO4 at 263.15 K is recorded as 0.0 S/m
                # (frozen / below detection); a percentage error is undefined.
                print(f'  {m:8.4f} {T_K:7.2f} {meas:9.4f} '
                      f'{"":>9} {"":>10} {"(skipped: no signal)":>19}')
                continue
            mc = sigma_total(compound, m, T_K)
            e_mc = 100 * (mc - meas) / meas
            if speciable:
                sp_val = sigma_speciated(compound, m, T_K)
                e_sp = 100 * (sp_val - meas) / meas
                ff = 100 * spec.free_fraction(compound, m, T_K)
                print(f'  {m:8.4f} {T_K:7.2f} {meas:9.4f} {mc:9.4f} '
                      f'{sp_val:10.4f} {e_mc:+8.1f}% {e_sp:+8.1f}% {ff:6.1f}%')
                rows.append((compound, e_mc, e_sp))
            else:
                print(f'  {m:8.4f} {T_K:7.2f} {meas:9.4f} {mc:9.4f} '
                      f'{"--":>10} {e_mc:+8.1f}% {"--":>9} {"--":>7}')
                rows.append((compound, e_mc, np.nan))
    return rows


def summarise(rows):
    if not rows:
        return
    print('\n' + '=' * 72)
    print('RMS error by compound')
    print(f'  {"compound":<10} {"MC12":>10} {"MC12+spec":>12}   verdict')
    df = pd.DataFrame(rows, columns=['compound', 'e_mc', 'e_sp'])
    for compound, g in df.groupby('compound'):
        rms_mc = float(np.sqrt(np.mean(np.square(g['e_mc']))))
        if g['e_sp'].notna().any():
            rms_sp = float(np.sqrt(np.nanmean(np.square(g['e_sp']))))
            if compound in NO_OP_COMPOUNDS:
                # Fully dissociated: expect agreement to well within 0.5%.
                # See check_no_op() for why this is not bit-exact.
                ok = abs(rms_sp - rms_mc) < 0.5
                verdict = ('unchanged, as required' if ok
                           else '*** REGRESSION: should be identical ***')
            elif rms_sp < rms_mc:
                verdict = f'improved {rms_mc / rms_sp:.1f}x'
            else:
                verdict = '*** WORSE -- check lambda parameters ***'
            print(f'  {compound:<10} {rms_mc:9.1f}% {rms_sp:11.1f}%   {verdict}')
        else:
            print(f'  {compound:<10} {rms_mc:9.1f}% {"--":>12}')


def check_no_op(tol=5e-3):
    """
    Speciation must not meaningfully change the fully-dissociated salts.

    Agreement is not bit-exact, and should not be: speciation additionally
    accounts for water's own dissociation (H+ and OH- at ~1.1e-7 molal, i.e.
    pH 7) and, for NH4Cl, slight NH4+ hydrolysis (~2.4e-6 molal H+ at 0.01
    molal). Both are real chemistry that the total-molality path omits. They
    move sigma by <0.05%, which matters most in relative terms at the lowest
    concentrations. The tolerance is set to 0.5% -- far below any experimental
    uncertainty, but tight enough to catch an actual association error.
    """
    print('\n' + '=' * 72)
    print(f'No-op check (WATEQ4F returns these fully dissociated; tol {tol:.1%})')
    worst = 0.0
    for compound in NO_OP_COMPOUNDS:
        if compound not in spec.SALT_RECIPES or compound not in ION_SPECS:
            continue
        for m in (0.01, 0.1, 1.0):
            a = sigma_total(compound, m, 298.15)
            b = sigma_speciated(compound, m, 298.15)
            rel = abs(b - a) / a
            worst = max(worst, rel)
            flag = 'ok' if rel < tol else '*** CHANGED ***'
            print(f'  {compound:<7} {m:5.2f} molal: '
                  f'{a:.6f} vs {b:.6f}  ({rel:.2e})  {flag}')
    print(f'  worst relative change: {worst:.2e} '
          f'({"pass" if worst < tol else "FAIL"})')
    return worst < tol


def check_lambda_sanity():
    """
    Flag implausible lambda_0 values for the charged complexes that speciation
    newly activates.

    Singly-charged complexes should have limiting molar conductivities near or
    below Cl- (~77 S cm2/mol at 25 C). NaSO4- and KSO4- in the current table are
    several times larger, which drives Na2SO4 conductivity *up* when speciated --
    physically backwards for an association correction. Until these are checked
    against MC12's published table, treat speciated sulfate/carbonate results as
    provisional.
    """
    print('\n' + '=' * 72)
    print('lambda_0(25 C) sanity check for speciation-activated complexes')
    ref = {}
    probe = {'Cl_m1': 'Cl-', 'HCO3_m1': 'HCO3-', 'HSO4_m1': 'HSO4-',
             'NaSO4_m1': 'NaSO4-', 'KSO4_m1': 'KSO4-', 'NaCO3_m1': 'NaCO3-'}
    for key, label in probe.items():
        ions = {key: {'mols': np.array([1e-9])}}
        # At infinite dilution lambda -> lambda_0; recover it as sigma/m.
        out = elecCondMcCleskey2012(25.0, ions)['sigma_Sm']
        lam0 = float(np.atleast_1d(out).ravel()[0]) / 1e-9 * 10.0
        ref[key] = lam0
    cl = ref['Cl_m1']
    for key, label in probe.items():
        ratio = ref[key] / cl
        flag = '' if ratio <= 1.5 else '  <-- implausible, verify vs MC12 table'
        print(f'  {label:<9} lambda_0 = {ref[key]:7.1f}  '
              f'({ratio:4.1f}x Cl-){flag}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['mahboub', 'cortes', 'both'],
                    default='mahboub')
    ap.add_argument('--temp', type=float, default=None,
                    help='Restrict to one temperature in K (e.g. 298.15)')
    ap.add_argument('--no-speciation', action='store_true',
                    help='Skip the speciated column (no Reaktoro needed)')
    args = ap.parse_args()

    use_spec = not args.no_speciation
    if use_spec and not spec.available():
        print('Reaktoro not importable in this interpreter -- speciated column '
              'disabled.\nRun with an environment that has reaktoro '
              '(e.g. ~/mamba/envs/PPcl/bin/python), or pass --no-speciation.\n')
        use_spec = False

    rows = []
    if args.dataset in ('mahboub', 'both'):
        print('=' * 72)
        print(f'Mahboub 2026 ({MAHBOUB_CSV})')
        df = load(MAHBOUB_CSV)
        if df is not None:
            rows += report(df, temp_K=args.temp, use_speciation=use_spec)
    if args.dataset in ('cortes', 'both'):
        print('\n' + '=' * 72)
        print(f'Cortes 2026 ({CORTES_CSV})')
        df = load(CORTES_CSV)
        if df is not None:
            rows += report(df, temp_K=args.temp, use_speciation=use_spec)

    summarise(rows)
    if use_spec:
        ok = check_no_op()
        check_lambda_sanity()
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
