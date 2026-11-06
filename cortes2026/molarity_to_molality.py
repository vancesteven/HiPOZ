#!/usr/bin/env python
"""
Convert Cortes et al. (2026) benchtop concentrations from molarity to molality.

The raw benchtop spreadsheet (JesusData2025.csv) reports concentrations in
mol per litre of solution ("0.5 M/L", "1.5:0.75 M/L"), and parse_benchtop_data
documents that it returns mol/L. Those values were nonetheless written to the
`concentration_molal` column of Cortes2026BenchtopData.csv unconverted, so the
McCleskey model was evaluated at concentrations 1-5% too low (more for
concentrated mixtures). This script corrects the column in place, keeping the
original mol/L value in a new `concentration_molar` column, and is idempotent:
it always recomputes from `concentration_molar` once that column exists.

Conversion: mol/L -> mol/kg H2O through the solution density computed by USGS
PHREEQC 3.8.6 with pitzer.dat (Pitzer activity model + Redlich-Rosenfeld molar
volumes, `density 1 calc`), at 20 C (solutions prepared at room temperature).
The resulting factors are stored in MOLALITY_TABLE below so that this script
needs no PHREEQC at run time; `--regenerate PATH_TO_PHREEQC` rebuilds the
table and prints it for pasting back here.

For mixtures, `concentration_molal` carries the FIRST component's
concentration (e.g. 'Na2SO4:KCl_2:1' at 1.5 means 1.5 M Na2SO4 + 0.75 M KCl);
the second component follows from the ratio in the compound name. The
molality/molarity factor is a property of the whole solution, so it applies
to both components alike. Organic solutes (amino acids) are not in pitzer.dat
and are left unconverted; NaCl+Glycine uses the NaCl factor (0.1 M glycine
neglected; no McCleskey comparison is made for it).

Usage:
    python cortes2026/molarity_to_molality.py              # rewrite the CSV
    python cortes2026/molarity_to_molality.py --dry-run    # show changes only
    python cortes2026/molarity_to_molality.py --regenerate /path/to/phreeqc
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'Cortes2026BenchtopData.csv')

# Salt -> {PHREEQC element: moles per mole of salt}
SALT_ELEMENTS = {
    'NaCl': {'Na': 1, 'Cl': 1},
    'KCl': {'K': 1, 'Cl': 1},
    'Na2SO4': {'Na': 2, 'S(6)': 1},
    'MgSO4': {'Mg': 1, 'S(6)': 1},
}

# (compound, first-component mol/L) -> first-component mol/kg H2O.
# Generated 2026-10-17 by --regenerate with PHREEQC 3.8.6 + pitzer.dat, 20 C.
MOLALITY_TABLE = {
    ('KCl', 0.5): 0.5080,
    ('KCl', 0.75): 0.7676,
    ('KCl', 1.0): 1.0310,
    ('KCl', 1.5): 1.5710,
    ('Na2SO4', 0.5): 0.5060,
    ('Na2SO4', 0.75): 0.7635,
    ('Na2SO4:KCl_1:1', 0.5): 0.5140,
    ('Na2SO4:KCl_1:1', 0.75): 0.7825,
    ('Na2SO4:KCl_1:1', 1.0): 1.0615,
    ('Na2SO4:KCl_1:2', 0.75): 0.8035,
    ('Na2SO4:KCl_2:1', 1.5): 1.6075,
    ('NaCl', 0.5): 0.5054,
    ('NaCl', 0.75): 0.7616,
    ('NaCl', 1.0): 1.0200,
    ('NaCl', 1.5): 1.5460,
    ('NaCl:MgSO4_1:1', 0.5): 0.5064,
    ('NaCl:MgSO4_1:1', 0.75): 0.7657,
    ('NaCl:MgSO4_1:1', 1.0): 1.0300,
    ('NaCl:MgSO4_1:1', 1.5): 1.5800,
    ('NaCl:MgSO4_1:2', 0.75): 0.7749,
    ('NaCl:MgSO4_2:1', 1.5): 1.5580,
}


def mixture_components(compound, first_molar):
    """
    Split a compound label into {salt: mol/L}.

    'Na2SO4:KCl_2:1' at 1.5 -> {'Na2SO4': 1.5, 'KCl': 0.75}. Single salts
    return {salt: first_molar}. Returns None for labels without a salt recipe.
    """
    if compound in SALT_ELEMENTS:
        return {compound: first_molar}
    m = re.fullmatch(r'(\w+):(\w+)_([\d.]+):([\d.]+)', compound)
    if m and m.group(1) in SALT_ELEMENTS and m.group(2) in SALT_ELEMENTS:
        r1, r2 = float(m.group(3)), float(m.group(4))
        return {m.group(1): first_molar, m.group(2): first_molar * r2 / r1}
    return None


def regenerate(phreeqc, database, compositions, T_C=20.0):
    """Run PHREEQC on each composition; return {(compound, molar): molal}."""
    deck = []
    for i, (compound, molar) in enumerate(compositions):
        comps = mixture_components(compound, molar)
        elements = {}
        for salt, c in comps.items():
            for el, n in SALT_ELEMENTS[salt].items():
                elements[el] = elements.get(el, 0.0) + n * c
        lines = '\n'.join(f'    {el} {amt:.6g}' for el, amt in elements.items())
        deck.append(f'SOLUTION {i + 1}\n    temp {T_C}\n    pH 7\n'
                    f'    units mol/l\n    density 1 calc\n{lines}\nEND')
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, 'in.pqi'), os.path.join(tmp, 'out.pqo')
        with open(inp, 'w') as f:
            f.write('\n'.join(deck))
        subprocess.run([phreeqc, inp, out, database], check=True,
                       capture_output=True)
        text = open(out).read()
    blocks = text.split('Beginning of initial solution calculations')[1:]
    table = {}
    for (compound, molar), block in zip(compositions, blocks):
        comp = block.split('Solution composition')[1].split('Description of solution')[0]
        molal = {m.group(1): float(m.group(2)) for m in
                 re.finditer(r'^\s+(\S+)\s+([\d.e+-]+)\s+[\d.e+-]+\s*$', comp, re.M)}
        first_salt = next(iter(mixture_components(compound, molar)))
        el, n = next(iter(SALT_ELEMENTS[first_salt].items()))
        el_key = el.split('(')[0]
        # Total molality of the first salt's first element over its input molarity
        # gives the solution's mol/kg-per-mol/L factor.
        input_molar = sum(n_ * c for s, c in mixture_components(compound, molar).items()
                          for e, n_ in SALT_ELEMENTS[s].items() if e == el)
        table[(compound, molar)] = round(molar * molal[el_key] / input_molar, 4)
    return table


def convert_frame(df):
    """Return a copy of df with corrected concentration_molal and a
    concentration_molar column holding the original mol/L values."""
    df = df.copy()
    if 'concentration_molar' not in df.columns:
        df.insert(df.columns.get_loc('concentration_molal'),
                  'concentration_molar', df['concentration_molal'])
    molal = []
    for compound, molar in zip(df['compound'], df['concentration_molar']):
        key = (compound, round(float(molar), 4))
        if key in MOLALITY_TABLE:
            molal.append(MOLALITY_TABLE[key])
        elif compound == 'NaCl+Glycine' and ('NaCl', key[1]) in MOLALITY_TABLE:
            molal.append(MOLALITY_TABLE[('NaCl', key[1])])
        else:
            molal.append(float(molar))  # organics: no density data; unchanged
    df['concentration_molal'] = molal
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default=CSV)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--regenerate', metavar='PHREEQC',
                   help='Path to a phreeqc executable; rebuild MOLALITY_TABLE')
    p.add_argument('--database', default=None,
                   help='pitzer.dat path (default: next to the phreeqc install)')
    args = p.parse_args(argv)

    header = []
    with open(args.csv) as f:
        for line in f:
            if line.startswith('#'):
                header.append(line)
            else:
                break
    df = pd.read_csv(args.csv, comment='#')

    if args.regenerate:
        db = args.database or os.path.join(
            os.path.dirname(os.path.dirname(args.regenerate)),
            'share', 'doc', 'phreeqc', 'database', 'pitzer.dat')
        molar_col = 'concentration_molar' if 'concentration_molar' in df else 'concentration_molal'
        comps = sorted({(c, round(float(m), 4)) for c, m in zip(df['compound'], df[molar_col])
                        if mixture_components(c, float(m)) is not None})
        table = regenerate(args.regenerate, db, comps)
        print('MOLALITY_TABLE = {')
        for k, v in table.items():
            print(f'    {k!r}: {v:.4f},')
        print('}')
        return 0

    out = convert_frame(df)
    changed = out[out['concentration_molal'] != out['concentration_molar']]
    summary = changed.groupby(['compound', 'concentration_molar'])['concentration_molal'].first()
    print(f'{len(changed)} of {len(out)} rows converted mol/L -> mol/kg:')
    print(summary.to_string())
    if args.dry_run:
        return 0
    with open(args.csv, 'w') as f:
        f.writelines(header)
        out.to_csv(f, index=False)
    print(f'Wrote {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
