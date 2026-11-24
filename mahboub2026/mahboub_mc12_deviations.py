#!/usr/bin/env python
"""
Deviations of the Mahboub et al. (2026) benchtop conductivities from the
McCleskey et al. (2012) model, before and after the 2026 corrections.

  pre-fix   per-ion ionic strength, full dissociation (variant A of
            cortes2026/compare_prepost_mccleskey.py; its `sigma(per_ion=True)`)
  corrected shared ionic strength, WATEQ4F speciation via Reaktoro
            (study_plots.compute_mccleskey_model with resolve_speciation 'auto',
            the path used by cortes2026/plot_mixture_families.py)

Delta % = 100 * (measured - model) / model (positive = data above model).
I_eff is the effective ionic strength of the speciated (free charged ions)
solution; in_range = I_eff <= 0.7 mol/kg and T >= 0 C.

The 'Mixture' rows (1:1:1 MgSO4:NaCl:Na2CO3) have no model (the concentration
basis is undefined and mixtures are not supported); they are excluded.

    python mahboub2026/mahboub_mc12_deviations.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, 'cortes2026')):
    sys.path.insert(0, p)

import speciation  # noqa: E402
import cortes_mccleskey as cm  # noqa: E402
from study_plots import compute_mccleskey_model, ION_SPECS  # noqa: E402
from compare_prepost_mccleskey import sigma as mc12_sigma  # noqa: E402

DATA = os.path.join(HERE, 'Mahboub2026BenchtopData.csv')
OUT_CSV = os.path.join(HERE, 'mahboub_mc12_deviations.csv')
OUT_TXT = os.path.join(HERE, 'mahboub_mc12_deviations_summary.txt')
I_MAX, T_MIN = 0.7, 0.0
Z = {'Na_p1': 1, 'K_p1': 1, 'NH4_p1': 1, 'Cl_m1': 1, 'Mg_p2': 2,
     'SO4_m2': 2, 'CO3_m2': 2}
COMPOUNDS = ['NaCl', 'MgSO4', 'NH4Cl', 'Na2CO3']


def prefix_model(compound, m, T_C):
    return mc12_sigma({k: v * m for k, v in ION_SPECS[compound].items()}, T_C, per_ion=True)


def corrected_model(compound, m, T_C, use_spec):
    out = compute_mccleskey_model(np.atleast_1d(m), np.atleast_1d(T_C) + 273.15,
                                  compound=compound, speciation=use_spec)
    return float(np.squeeze(out))


def effective_I(compound, m, T_C, use_spec):
    """Ionic strength (mol/kg) of the free charged ions; None-fallback aware."""
    T_K = T_C + 273.15
    free = speciation.free_ion_molalities(compound, m, T_K) if use_spec else None
    converged = free is not None or not use_spec
    if free is None:
        free = {k: v * m for k, v in ION_SPECS[compound].items()}
    I = 0.5 * sum(v * Z[k] ** 2 for k, v in free.items() if k in Z)
    return I, converged


def compute(df=None):
    if df is None:
        df = pd.read_csv(DATA, comment='#')
    df = df[df.compound.isin(COMPOUNDS)].reset_index(drop=True)
    use_spec = {c: cm.resolve_speciation(c, 'auto', ion_spec=ION_SPECS[c])
                for c in COMPOUNDS}
    rows, notes = [], []
    cache = {}
    for _, r in df.iterrows():
        c, m, T = r.compound, float(r.concentration_molal), float(r.temperature_C)
        key = (c, m, T)
        if key not in cache:
            pre = prefix_model(c, m, T)
            post = corrected_model(c, m, T, use_spec[c])
            I, conv = effective_I(c, m, T, use_spec[c])
            if not conv:
                notes.append(f'speciation did not converge: {key} (stoichiometric fallback)')
            cache[key] = (pre, post, I)
        pre, post, I = cache[key]
        meas = float(r.conductivity_Sm)
        rows.append(dict(
            compound=c, molal=m, T_C=T, replicate=r.replicate, meas_Sm=meas,
            model_prefix_Sm=pre, model_corrected_Sm=post,
            delta_pre_pct=100 * (meas - pre) / pre,
            delta_post_pct=100 * (meas - post) / post,
            I_eff=I, in_range=bool(I <= I_MAX and T >= T_MIN)))
    return pd.DataFrame(rows), use_spec, sorted(set(notes))


def summary_text(res, use_spec, notes):
    L = ['McCleskey (2012) model vs Mahboub et al. (2026) benchtop data',
         'Delta % = 100*(meas - model)/model.  RMS and mean of Delta %.',
         'in-range: I_eff <= 0.7 mol/kg and T >= 0 C.  Mixture rows: no model, excluded.',
         'Speciation (resolve_speciation auto): ' +
         ', '.join(f'{c}={use_spec[c]}' for c in COMPOUNDS), '']
    hdr = (f'{"compound":8s} {"subset":9s} {"n":>4s} {"RMS pre":>8s} {"RMS post":>9s} '
           f'{"mean pre":>9s} {"mean post":>10s}')
    L.append(hdr)
    for c in COMPOUNDS:
        for label, s in (('all', res[res.compound == c]),
                         ('in-range', res[(res.compound == c) & res.in_range])):
            if len(s) == 0:
                L.append(f'{c:8s} {label:9s} {0:4d}')
                continue
            rms = lambda x: np.sqrt((x ** 2).mean())
            L.append(f'{c:8s} {label:9s} {len(s):4d} {rms(s.delta_pre_pct):8.2f} '
                     f'{rms(s.delta_post_pct):9.2f} {s.delta_pre_pct.mean():+9.2f} '
                     f'{s.delta_post_pct.mean():+10.2f}')
    L += [''] + (notes or ['All speciation solves converged.'])
    return '\n'.join(L) + '\n'


def main():
    res, use_spec, notes = compute()
    res.to_csv(OUT_CSV, index=False, float_format='%.6g')
    txt = summary_text(res, use_spec, notes)
    open(OUT_TXT, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
