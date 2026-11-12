#!/usr/bin/env python3
"""
Concentration at which each Cortes solution reaches McCleskey et al. (2012)'s
upper validated effective ionic strength, I_eff = 0.7 mol/kg.

McCleskey et al. (2012, GCA 77:369) validate their conductivity method for
effective ionic strength 0.0004-0.7 mol/kg, with I_eff taken from a WATEQ4F
speciation. Ion pairing (NaSO4-, KSO4-, MgSO4 aq) lowers I_eff below the
nominal value, so the limit for sulfate-bearing solutions lies above the
nominal 0.7/I_per_mol. Chlorides have no WATEQ4F ion pairs, so their limit is
exactly 0.7 mol/kg.

The table in cortes_mccleskey.MCCLESKEY_LIMITS was generated with

    python cortes2026/mccleskey_validity_limits.py --phreeqc /path/to/phreeqc

(USGS PHREEQC 3.8.6, wateq4f.dat; PHREEQC's reported ionic strength at pH 7,
interpolated on a 0.005 mol/kg grid). Concentration is the molality of the
FIRST component, matching get_mixture_ion_spec. The table stores the 5 C value,
the most restrictive across the benchtop temperatures (5, 10, 20 C).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cortes_mccleskey import ION_SPECS, get_mixture_ion_spec  # noqa: E402

I_MAX = 0.7
TEMPS_C = (5, 10, 20, 25)
COMPOUNDS = ['NaCl', 'KCl', 'Na2SO4', 'MgSO4',
             'Na2SO4:KCl_1:1', 'Na2SO4:KCl_1:2', 'Na2SO4:KCl_2:1',
             'NaCl:MgSO4_1:1', 'NaCl:MgSO4_1:2', 'NaCl:MgSO4_2:1']
PHREEQC_ELEMENT = {'Na_p1': 'Na', 'K_p1': 'K', 'Mg_p2': 'Mg',
                   'Cl_m1': 'Cl', 'SO4_m2': 'S(6)'}


def ion_spec_for(compound):
    return get_mixture_ion_spec(compound) or ION_SPECS[compound]


def ionic_strengths(phreeqc, database, spec, molals, T_C):
    """PHREEQC effective ionic strength for each first-component molality."""
    deck = []
    for i, m in enumerate(molals):
        totals = {}
        for ion, mult in spec.items():
            el = PHREEQC_ELEMENT[ion]
            totals[el] = totals.get(el, 0.0) + mult * m
        lines = '\n'.join(f'    {el} {v:.6g}' for el, v in totals.items())
        deck.append(f'SOLUTION {i + 1}\n    temp {T_C}\n    pH 7\n'
                    f'    units mol/kgw\n{lines}\nEND')
    with tempfile.TemporaryDirectory() as tmp:
        pqi, out = os.path.join(tmp, 'in.pqi'), os.path.join(tmp, 'out.txt')
        with open(pqi, 'w') as f:
            f.write('\n'.join(deck))
        subprocess.run([phreeqc, pqi, out, database], capture_output=True,
                       check=True, cwd=tmp)
        text = open(out).read()
    I = [float(x) for x in
         re.findall(r'Ionic strength \(mol/kgw\)\s+=\s+([\d.eE+-]+)', text)]
    return np.array(I[:len(molals)])


def limit_molal(phreeqc, database, compound, T_C):
    spec = ion_spec_for(compound)
    # PHREEQC stops converging for MgSO4 near 0.69 m; the limit is far below
    top = 0.6 if 'Mg_p2' in spec else 1.5
    grid = np.round(np.arange(0.02, top + 1e-9, 0.005), 4)
    I = ionic_strengths(phreeqc, database, spec, grid, T_C)
    if len(I) != len(grid) or not (I.min() < I_MAX < I.max()):
        raise RuntimeError(f'{compound} at {T_C} C: I_eff={I_MAX} not bracketed')
    return float(np.interp(I_MAX, I, grid))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--phreeqc', required=True, help='PHREEQC executable')
    p.add_argument('--database', help='wateq4f.dat (default: next to the '
                   'executable, ../../database/wateq4f.dat)')
    a = p.parse_args()
    db = a.database or os.path.join(os.path.dirname(a.phreeqc), '..', '..',
                                    'database', 'wateq4f.dat')
    print(f"{'compound':16s} " + ' '.join(f'{T:>4d}C ' for T in TEMPS_C))
    for c in COMPOUNDS:
        vals = [limit_molal(a.phreeqc, db, c, T) for T in TEMPS_C]
        print(f'{c:16s} ' + ' '.join(f'{v:.4f}' for v in vals))


if __name__ == '__main__':
    main()
