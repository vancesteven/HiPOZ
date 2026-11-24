#!/usr/bin/env python3
"""
One figure per mixture family, all ratios and concentrations together.

Families: NaCl:MgSO4 (1:1, 1:2, 2:1) and Na2SO4:KCl (1:1, 1:2, 2:1), from the
Cortes benchtop probe data (Cortes2026BenchtopData.csv). For each family:

  <family>_all_vs_temperature.pdf    sigma vs T, one line per (ratio, molality)
  <family>_all_vs_concentration.pdf  sigma vs first-component molality, one
                                     line per (ratio, temperature)

Colour encodes the ratio; within a ratio, the marker encodes the molality
(vs T) or the line style the temperature (vs m). McCleskey (2012) curves use
the same ion specs and speciation as the dedicated per-mixture figures
(cortes2026_plots.py), drawn dashed; error bars follow the same rule (3 % of
sigma for single readings). These complement, and do not replace, the
dedicated figures.

Usage:  python cortes2026/plot_mixture_families.py
Writes: cortes2026/cortes_plots/mixture_families/
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.chdir(REPO)
import cortes_mccleskey as cm  # noqa: E402
from plotting import DELTA_LABEL, pdiff  # noqa: E402
from study_plots import compute_mccleskey_model  # noqa: E402

DATA = os.path.join(HERE, 'Cortes2026BenchtopData.csv')
OUT = os.path.join(HERE, 'cortes_plots', 'mixture_families')

FAMILIES = {
    'NaCl:MgSO4': (r'NaCl:MgSO$_4$', 'NaCl'),
    'Na2SO4:KCl': (r'Na$_2$SO$_4$:KCl', r'Na$_2$SO$_4$'),
}
# Ratio colours (dataviz validate_palette, light surface: all checks pass)
RATIO_COLORS = {'1:1': '#184f95', '2:1': '#eda100', '1:2': '#a8322f'}
MOLAL_MARKERS = {0.5: 'o', 0.75: 's', 1.0: '^', 1.5: 'D'}
TEMP_STYLES = {5: ':', 10: '--', 20: '-'}
SINGLE_READING_REL_ERR = 0.03  # as study_plots for n = 1
DELTA_ERR_PCT = 5.0  # Delta error bar, as the per-mixture figures (study_plots)


def load():
    d = pd.read_csv(DATA, comment='#')
    d = d[d.compound.str.contains(':', na=False)].copy()
    d['family'] = d.compound.str.split('_').str[0]
    d['ratio'] = d.compound.str.split('_').str[1]
    g = (d.groupby(['family', 'ratio', 'compound', 'concentration_molal', 'temperature_C'])
          .conductivity_Sm.agg(['mean', 'std', 'size']).reset_index())
    g['err'] = np.where(g['size'] > 1, g['std'], SINGLE_READING_REL_ERR * g['mean'])
    return g


def model(compound, molal, T_C):
    spec = cm.get_mixture_ion_spec(compound)
    sp = cm.resolve_speciation(compound, 'auto', ion_spec=spec)
    out = compute_mccleskey_model(np.atleast_1d(molal), np.atleast_1d(T_C) + 273.15,
                                  compound=compound, ion_spec=spec, speciation=sp)
    return np.array([o for o in out]).squeeze()


def _style():
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
        'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    })


def _label(ax, x, y, text, col):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points', va='center',
                fontsize=7.8, color='#0b0b0b', zorder=5)


def _two_panels():
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(7.6, 8.6), sharex=True,
                                  gridspec_kw={'height_ratios': [1.6, 1]})
    for a in (ax, axd):
        a.grid(True, color='#e4e3df', lw=0.6)
        a.set_axisbelow(True)
    axd.axhline(0, color='#9a9a9a', lw=0.8, zorder=1)
    axd.set_ylabel(DELTA_LABEL)
    return fig, ax, axd


RANGE_NOTE = (r'every point is beyond the McCleskey (2012) validated range '
              r'($I_\mathrm{eff} > 0.7$ mol/kg)')


def _spread_labels(ax, items, gap_frac=0.045):
    """Right-of-point labels, nudged apart vertically where they would overlap.

    items: (x, y, text) in data coordinates."""
    lo, hi = ax.get_ylim()
    gap = gap_frac * (hi - lo)
    placed = []
    for x, y, text in sorted(items, key=lambda it: it[1]):
        yt = y
        for xp, yp in placed:
            if abs(xp - x) < 1e-9 and yt - yp < gap:
                yt = yp + gap
        placed.append((x, yt))
        # x offset in points, y offset converted from the data shift to points
        dy_pts = (ax.transData.transform((x, yt))[1] - ax.transData.transform((x, y))[1]) \
            * 72 / ax.figure.dpi
        ax.annotate(text, (x, y), xytext=(8, dy_pts), textcoords='offset points',
                    va='center', fontsize=7.8, color='#0b0b0b', zorder=5)


def plot_vs_T(g, fam, path):
    name, first = FAMILIES[fam]
    fig, ax, axd = _two_panels()
    sub = g[g.family == fam]
    Tm = np.linspace(4, 21, 40)
    top, bottom = [], []
    for (ratio, compound, m), r in sub.groupby(['ratio', 'compound', 'concentration_molal']):
        col, mk = RATIO_COLORS[ratio], MOLAL_MARKERS[m]
        r = r.sort_values('temperature_C')
        ax.plot(Tm, [float(model(compound, m, t)) for t in Tm], '--', color=col, lw=0.9,
                alpha=0.7, zorder=1)
        ax.errorbar(r.temperature_C, r['mean'], yerr=r.err, fmt='-' + mk, color=col, ms=6,
                    mec='#fcfcfb', mew=1, lw=1.4, elinewidth=1, capsize=0, zorder=3)
        d = pdiff(r['mean'].to_numpy(), [float(model(compound, m, t)) for t in r.temperature_C])
        axd.errorbar(r.temperature_C, d, yerr=DELTA_ERR_PCT, fmt='-' + mk, color=col, ms=6,
                     mec='#fcfcfb', mew=1, lw=1.2, elinewidth=1, capsize=0, zorder=3)
        top.append((r.temperature_C.iloc[-1], r['mean'].iloc[-1], f'{ratio}, {m:g} m'))
        bottom.append((r.temperature_C.iloc[-1], d[-1], f'{ratio}, {m:g} m'))
    axd.set_xlabel(r'Temperature ($^\circ$C)')
    ax.set_ylabel(r'$\sigma$ (S/m)')
    ax.set_xlim(3, 24.5)
    _spread_labels(ax, top)
    _spread_labels(axd, bottom)
    axd.text(0.01, 0.03, RANGE_NOTE, transform=axd.transAxes, fontsize=7.5, color='#52514e')
    ax.set_title(f'{name} mixtures, all ratios and concentrations (benchtop probe)', fontsize=11)
    ax.legend([Line2D([], [], ls='--', color='#52514e', lw=0.9)],
              ['McCleskey et al. (2012), same composition'],
              fontsize=8, frameon=False, loc='upper left')
    ax.text(0.99, 0.02, f'labels: {name} ratio, {first} molality (mol/kg)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path[:-4] + '.png', dpi=200)
    plt.close(fig)


def plot_vs_m(g, fam, path):
    name, first = FAMILIES[fam]
    fig, ax, axd = _two_panels()
    sub = g[g.family == fam]
    top, bottom = [], []
    for (ratio, compound, T), r in sub.groupby(['ratio', 'compound', 'temperature_C']):
        col, ls = RATIO_COLORS[ratio], TEMP_STYLES[int(T)]
        r = r.sort_values('concentration_molal')
        mm = np.linspace(0.4, r.concentration_molal.max() + 0.05, 40)
        if len(r) > 1:
            ax.plot(mm, model(compound, mm, T), ls, color=col, lw=0.8, alpha=0.55, zorder=1)
        else:  # one concentration only: the model as a hollow point beside it
            ax.plot(r.concentration_molal, [float(model(compound, r.concentration_molal.iloc[0], T))],
                    'o', mfc='none', mec=col, ms=6, mew=1.1, zorder=2)
        ax.errorbar(r.concentration_molal, r['mean'], yerr=r.err, fmt='o', color=col, ms=6,
                    mec='#fcfcfb', mew=1, elinewidth=1, capsize=0, zorder=3)
        ax.plot(r.concentration_molal, r['mean'], ls, color=col, lw=1.6, zorder=2)
        d = pdiff(r['mean'].to_numpy(), [float(model(compound, m, T)) for m in r.concentration_molal])
        axd.errorbar(r.concentration_molal, d, yerr=DELTA_ERR_PCT, fmt='o', color=col, ms=6,
                     mec='#fcfcfb', mew=1, elinewidth=1, capsize=0, zorder=3)
        axd.plot(r.concentration_molal, d, ls, color=col, lw=1.4, zorder=2)
        lbl = fr'{ratio}, {T:g} $^\circ$C'
        top.append((r.concentration_molal.iloc[-1], r['mean'].iloc[-1], lbl))
        bottom.append((r.concentration_molal.iloc[-1], d[-1], lbl))
    # McCleskey validated range ends at a different first-component molality
    # for each ratio: one dashed line per ratio, in its colour
    for ratio in sorted(sub.ratio.unique()):
        lim = cm.get_mccleskey_limit(f'{fam}_{ratio}')
        for a in (ax, axd):
            a.axvline(lim, ls=(0, (4, 3)), color=RATIO_COLORS[ratio], lw=1.4, alpha=0.8, zorder=1)
        axd.annotate(f'MC12 limit {ratio}', (lim, 1), xycoords=('data', 'axes fraction'),
                     xytext=(3, -4), textcoords='offset points', rotation=90, va='top',
                     fontsize=7, color='#52514e')
    axd.set_xlabel(f'{first} molality (mol/kg)')
    ax.set_ylabel(r'$\sigma$ (S/m)')
    ax.set_xlim(0.1, 1.85)
    _spread_labels(ax, top)
    _spread_labels(axd, bottom)
    axd.text(0.99, 0.03, RANGE_NOTE, transform=axd.transAxes, ha='right', fontsize=7.5,
             color='#52514e')
    ax.set_title(f'{name} mixtures, all ratios and temperatures (benchtop probe)', fontsize=11)
    key = [Line2D([], [], ls=TEMP_STYLES[t], color='#52514e', lw=1.6) for t in (5, 10, 20)]
    key += [Line2D([], [], ls=TEMP_STYLES[20], color='#52514e', lw=0.8, alpha=0.55),
            Line2D([], [], ls='', marker='o', mfc='none', mec='#52514e', ms=6),
            Line2D([], [], ls=(0, (4, 3)), color='#52514e', lw=1.4)]
    ax.legend(key, [r'5 $^\circ$C', r'10 $^\circ$C', r'20 $^\circ$C',
                    'thin: McCleskey et al. (2012)', 'hollow: McCleskey, single concentration',
                    'dashed vertical: McCleskey limit for each ratio'],
              fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(0.2, 1.0))
    ax.text(0.99, 0.02, f'labels: {name} ratio, temperature',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color='#52514e')
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path[:-4] + '.png', dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    _style()
    g = load()
    for fam in FAMILIES:
        stem = fam.replace(':', '-').lower()
        plot_vs_T(g, fam, os.path.join(OUT, f'{stem}_all_vs_temperature.pdf'))
        plot_vs_m(g, fam, os.path.join(OUT, f'{stem}_all_vs_concentration.pdf'))
        print('wrote', stem)


if __name__ == '__main__':
    main()
