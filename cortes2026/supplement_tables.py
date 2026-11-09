#!/usr/bin/env python3
"""
Supplement tables (LaTeX) for the Cortes, Diaz and Garrity conductivity data.

One .tex file per author, each holding longtables:

  cortes2026/supplement/cortes_tables.tex
      S-C1  Gamry high-P sweeps, Aug 13-20 2025, 0.5 mol/kg NaCl (every sweep)
      S-C2  the same, step means (as plotted)
      S-C3  benchtop probe data (Cortes2026BenchtopData.csv)
  cortes2026/supplement/diaz_tables.tex
      S-D1  Gamry high-P sweeps, Oct 6-Nov 6 2026 (every sweep)
      S-D2  the same, step means (as plotted)
      S-D3  1-bar sweeps, Sep 17-25 2026, 0.5 mol/kg NaCl (every sweep)
  cortes2026/supplement/garrity_tables.tex
      S-G1  Gamry 1-bar sweeps, Feb 19 2026 (every sweep)
      S-G2  probe readings from her notebook

Values are written exactly as stored in each source file (the string, not a
re-formatted float), so every table can be checked against its source with
check_supplement_tables.py. Sweeps and steps left out of the figures stay in
the tables, flagged in the last column. Step means come from
plot_nacl_sigma_P.py, which must be run first.

Usage:  python cortes2026/supplement_tables.py
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'supplement')

CORTES_CONFIGS = ['data/20250813Cortes/zAnalysis20250813.csv',
                  'data/20250814Cortes/zAnalysis20250814.csv',
                  'data/20250818Cortes/zAnalysis20250818.csv',
                  'data/20250819Cortes/zAnalysis20250819Cortes.csv',
                  'data/20250820Cortes/zAnalysis20250820Cortes.csv']
DIAZ_1BAR_CONFIGS = ['data/20260917/zAnalysis20260917.csv',
                     'data/20260923/zAnalysis20260923.csv',
                     'data/20260924/zAnalysis20260924.csv',
                     'data/20260925/zAnalysis20260925.csv']
GARRITY_CONFIG = 'data/20260219/zAnalysis20260219.csv'
BENCHTOP = 'cortes2026/Cortes2026BenchtopData.csv'
GARRITY_PROBE = 'cortes2026/garrity/garrity_probe_readings.csv'
DIAZ_HP_SWEEPS = 'cortes2026/diaz/diaz_hp_sweep_fits.csv'
CORTES_STEPS = 'cortes2026/nacl_sigma_P/nacl_0p5m_sigma_P_steps.csv'
DIAZ_STEPS = 'cortes2026/nacl_sigma_P/sigma_P_2026_steps.csv'


def read_str(path, **kw):
    return pd.read_csv(os.path.join(REPO, path), dtype=str, keep_default_na=False, **kw)


def tex(s):
    """Escape a source string for LaTeX without changing its characters."""
    s = str(s)
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}')):
        s = s.replace(a, b)
    return s.replace('µ', r'\textmu{}')


def fname(s):
    """A filename that may break at any character (underscores give no breaks)."""
    return r'\texttt{' + r'\allowbreak{}'.join(tex(c) for c in str(s)) + '}'


def fmt(x, nd):
    """Format a derived value (step means only)."""
    return '' if pd.isna(x) or x == '' else f'{float(x):.{nd}f}'


def longtable(caption, label, header, rows, colspec, source):
    lines = [r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\setlength{\LTcapwidth}{\textwidth}',
             r'\begin{longtable}{' + colspec + '}',
             r'\caption{' + caption + ' Source: ' + ', '.join(r'\texttt{' + tex(x.strip()) + '}'
                                                         for x in source.split(','))
             + r'.}\label{' + label + r'}\\',
             r'\toprule', ' & '.join(header) + r' \\', r'\midrule', r'\endfirsthead',
             r'\multicolumn{' + str(len(header)) + r'}{l}{\emph{(continued)}}\\',
             r'\toprule', ' & '.join(header) + r' \\', r'\midrule', r'\endhead',
             r'\bottomrule', r'\endfoot']
    lines += [' & '.join(r) + r' \\' for r in rows]
    lines += [r'\end{longtable}', r'\endgroup', '']
    return '\n'.join(lines)


def _flag(row):
    ex = row.get('exclude', '')
    if ex and ex.lower() in ('x', 'true', '1'):
        m = re.search(r'excluded: ([^;"]+)', row.get('notes', ''))
        return 'excluded' + (f': {m.group(1).strip()}' if m else '')
    return ''


def config_sweep_rows(paths, keep=lambda r: True, note=lambda r: ''):
    rows = []
    for p in paths:
        d = read_str(p)
        for _, r in d.iterrows():
            if not keep(r):
                continue
            flag = '; '.join(x for x in (_flag(r), note(r)) if x)
            rows.append([fname(r.filename), tex(r.type), tex(r.P_MPa), tex(r.T_K), tex(r.get('Z_Ohm', '')),
                         tex(r.get('Z_unc_Ohm', '')), tex(r.conductivity_Sm), tex(r.comp),
                         tex(r.w_molal), tex(flag)])
    return rows


SWEEP_HEADER = ['File', 'Type', r'$P$ (MPa)', r'$T$ (K)', r'$R$ ($\Omega$)', r'$\delta R$ ($\Omega$)',
                r'$\sigma$ (S/m)', 'Solute', r'$m$ (mol/kg)', 'Flag']
SWEEP_COLS = r'p{5.2cm}lrrrrrlrp{4.2cm}'


def cortes_tables():
    excluded_steps = read_str(CORTES_STEPS)
    excl = {}
    for _, s in excluded_steps[excluded_steps.excluded != ''].iterrows():
        excl[s.start] = s.excluded
    parts = []
    # S-C1: every sweep, as stored in the zAnalysis configs. The figure's
    # exclusions are per step; the step's first sweep time is the key.
    parts.append(longtable(
        'Cortes, Gamry high-pressure cell, Aug 13--20 2025: every sweep as stored in the '
        'zAnalysis configs. $R$ is the CPE-circuit resistance (10--100 kHz), $\\sigma = K/R$ '
        'with $K$ from the 8 S/m KCl standard of Aug 18 (Aug 19--20 borrow it). Standards '
        'and the 0.75 and 2 mol/kg and NaCl:MgSO$_4$ bench sweeps of Aug 14 are included. '
        'Steps left out of the $\\sigma(P)$ figure are flagged in Table~\\ref{tab:cortes-steps}.',
        'tab:cortes-sweeps', SWEEP_HEADER,
        config_sweep_rows(CORTES_CONFIGS), SWEEP_COLS, ', '.join(CORTES_CONFIGS)))
    # S-C2: step means
    rows = []
    for _, s in excluded_steps.iterrows():
        rows.append([tex(s.start), s.n, fmt(s.P_MPa, 1), fmt(s.T_C, 2), fmt(s.sigma_Sm, 3),
                     fmt(s.sigma_min, 3), fmt(s.sigma_max, 3), fmt(s.drift_pct, 2), tex(s.excluded)])
    parts.append(longtable(
        'Cortes, 0.5 mol/kg NaCl: step means of the sweeps in Table~\\ref{tab:cortes-sweeps} '
        '(consecutive sweeps $<$4 min apart at the same nominal $P$), as plotted. Drift is '
        'the change from the first to the last sweep of a step. Flagged steps are left out '
        'of the figure.',
        'tab:cortes-steps',
        ['Start', '$n$', r'$P$ (MPa)', r'$T$ ($^\circ$C)', r'$\bar\sigma$ (S/m)', r'$\sigma_\mathrm{min}$',
         r'$\sigma_\mathrm{max}$', r'Drift (\%)', 'Left out (reason)'],
        rows, r'lrrrrrrrp{5cm}', CORTES_STEPS))
    # S-C3: benchtop probe data
    b = read_str(BENCHTOP, comment='#')
    rows = [[tex(r.compound), tex(r.concentration_molal), tex(r.temperature_C), tex(r.conductivity_Sm),
             tex(r.replicate), tex(r.notes)] for _, r in b.iterrows()]
    parts.append(longtable(
        'Cortes, benchtop conductivity probe. Concentrations are molalities as prepared '
        '(labels "M/L" are mol/kg; the lab prepares only molal solutions). Mixtures give the '
        'first component\'s molality and the ratio in the name.',
        'tab:cortes-benchtop',
        ['Solute', r'$m$ (mol/kg)', r'$T$ ($^\circ$C)', r'$\sigma$ (S/m)', 'Rep.', 'Label'],
        rows, r'lrrrrp{6cm}', BENCHTOP))
    return '\n'.join(parts)


def diaz_tables():
    parts = []
    sw = read_str(DIAZ_HP_SWEEPS)
    rows = [[fname(r.filename), tex(r.run), tex(r.P_MPa), tex(r.T_K), tex(r.R), tex(r.R_unc),
             tex(r.K_cell), tex(r.sigma_Sm), tex(r.excluded)] for _, r in sw.iterrows()]
    parts.append(longtable(
        'Diaz, Gamry high-pressure cell, Oct 6--Nov 6 2026: every sweep, refitted from the raw '
        'Gamry files (CPE circuit, 10--100 kHz). $\\sigma = K/R$; $K$ = 8 S/m $\\times$ $R$ of '
        'the first three ambient sweeps of the standard run (Oct 14 for the Oct NaCl runs, '
        'which had no standard of their own: an estimate). Flagged sweeps are left out of the figure.',
        'tab:diaz-sweeps',
        ['File', 'Run', r'$P$ (MPa)', r'$T$ (K)', r'$R$ ($\Omega$)', r'$\delta R$ ($\Omega$)',
         r'$K$ (m$^{-1}$)', r'$\sigma$ (S/m)', 'Flag'],
        rows, r'p{4.1cm}p{2.6cm}rrrrrrp{2.8cm}', DIAZ_HP_SWEEPS))
    st = read_str(DIAZ_STEPS)
    rows = [[tex(r.run), tex(r.time[:16]), r.n, fmt(r.P_MPa, 1), fmt(r.T_C, 2), fmt(r.sigma_Sm, 4),
             fmt(r.sigma_lo, 4), fmt(r.sigma_hi, 4), 'down' if r.descent == 'True' else 'up']
            for _, r in st.iterrows()]
    parts.append(longtable(
        'Diaz, high-pressure step means of the unflagged sweeps in Table~\\ref{tab:diaz-sweeps}, as plotted.',
        'tab:diaz-steps',
        ['Run', 'Start', '$n$', r'$P$ (MPa)', r'$T$ ($^\circ$C)', r'$\bar\sigma$ (S/m)',
         r'$\sigma_\mathrm{min}$', r'$\sigma_\mathrm{max}$', 'Leg'],
        rows, r'p{4.2cm}lrrrrrrl', DIAZ_STEPS))
    parts.append(longtable(
        'Diaz, 1-bar sweeps of 0.5 mol/kg NaCl, Sep 17--25 2026, as stored in the zAnalysis '
        'configs (imported from his workbooks). $T$ is the bath setpoint.',
        'tab:diaz-1bar', SWEEP_HEADER, config_sweep_rows(DIAZ_1BAR_CONFIGS), SWEEP_COLS,
        ', '.join(DIAZ_1BAR_CONFIGS)))
    return '\n'.join(parts)


def garrity_tables():
    parts = []
    parts.append(longtable(
        'Garrity, Gamry sweeps at 1 bar, Feb 19 2026, as stored in the zAnalysis config. '
        'Labels "M" are mol/kg. The cell constant is provisional (the 15 mS/cm and 2764 '
        '$\\mu$S/cm standards give $K$ 3\\% apart).',
        'tab:garrity-gamry', SWEEP_HEADER, config_sweep_rows([GARRITY_CONFIG]), SWEEP_COLS,
        GARRITY_CONFIG))
    g = read_str(GARRITY_PROBE, comment='#')
    rows = [[tex(r.page), tex(r.date), tex(r.solute), tex(r.concentration), tex(r.g_MgSO4_per_10mL),
             tex(r.reading), tex(r.value), tex(r.reading_units), tex(r.T_C or r.T_C_written),
             tex(r.notes)] for _, r in g.iterrows()]
    parts.append(longtable(
        'Garrity, benchtop probe readings as written in her notebook (pp.~1, 3, 5, 6). '
        'Readings are unconverted. Room-temperature readings (pp.~1, 3) were probably '
        'taken with the probe\'s temperature compensation on (p.~4).',
        'tab:garrity-probe',
        ['p.', 'Date', 'Solute', r'$m$ (mol/kg)', r'g MgSO$_4$/10 mL', r'\#', 'Reading', 'Units',
         r'$T$ ($^\circ$C)', 'Note'],
        rows, r'rllllrrlp{1.8cm}p{3.6cm}', GARRITY_PROBE))
    return '\n'.join(parts)


HEADER = ('% Generated by cortes2026/supplement_tables.py; do not edit by hand.\n'
          '% Needs \\usepackage{longtable,booktabs}. Values copied verbatim from the source files.\n\n')


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (('cortes', cortes_tables), ('diaz', diaz_tables), ('garrity', garrity_tables)):
        path = os.path.join(OUT, f'{name}_tables.tex')
        with open(path, 'w') as f:
            f.write(HEADER + fn())
        print('wrote', os.path.relpath(path, REPO))


if __name__ == '__main__':
    main()
