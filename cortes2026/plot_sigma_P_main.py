#!/usr/bin/env python3
"""
MAIN-TEXT figure: best assessment of sigma(P, T) at each concentration.

Purpose
    One panel per solution (NaCl 0.5, NaCl 0.9, KCl 0.70 mol/kg): absolute
    conductivity vs pressure, colour = temperature, with a smooth guide through
    each isotherm. Only equilibrated runs are shown; every diagnostic (excluded
    steps, rejected runs, literature of other concentrations) lives in the
    supplement figure (plot_nacl_sigma_P.py).

Inputs   CSVs written by plot_nacl_sigma_P.py (read through sigma_P_sources.py).
         No sweep fitting here: runs in seconds.
Outputs  nacl_sigma_P/sigma_P_main.{pdf,png}
         nacl_sigma_P/sigma_P_main_anchor_factors.csv (also printed)
Usage    python cortes2026/plot_sigma_P_main.py

Policy (edit the constants below; each is a decision for PI review)
    * Runs: MAIN_RUNS. Not in the main text: Oct 6, Oct 9 (10-min steps),
      Oct 14-15 (leak during the run), any excluded or provisional step, and
      the Aug-Sep 2026 runs. Fisher and Fox / Fisher 1962 have no matching
      concentration (0.0005-0.02 M KCl vs our 0.70 m), so they are left out.
    * ANCHOR: each run is scaled so its first ascending 1-bar step equals the
      reference 1-bar conductivity (McCleskey 2011 measurements where
      tabulated, else McCleskey et al. 2012; sigma_P_sources.sigma_ref_1bar).
      The MEASURED pressure dependence is kept; the run's own absolute
      calibration is replaced by prior data.
    * Error bars: step min-max range combined in quadrature with the anchor
      uncertainty (ANCHOR_UNC, relative). ASSUMPTIONS, not derived.
    * Guide: a polynomial of P fitted to each isotherm's anchored points
      (compression and decompression both), drawn only between the lowest and
      highest measured pressure, so there is no extrapolation. The degree grows
      with the number of points (GUIDE_DEGREE). It is a visual guide, not a
      model; a cubic captures the single maximum of sigma(P) without wiggles.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigma_P_sources as src  # noqa: E402
OUT = src.OUT

# Runs shown (run keys of sigma_P_sources). Equilibrated runs only.
MAIN_RUNS = [
    # Aug 2025 0.5 mol/kg NaCl isotherms, kept steps only (Aug 14: all excluded)
    'NaCl 0.5 m, Aug 13, 20 C', 'NaCl 0.5 m, Aug 18, 20 C', 'NaCl 0.5 m, Aug 19, 0 C',
    'NaCl 0.5 m, Aug 19, 10 C', 'NaCl 0.5 m, Aug 20, 20 C', 'NaCl 0.5 m, Aug 20, 35 C',
    # 0.9 mol/kg NaCl: Oct 7-8 (30-min steps), Nov 16
    'NaCl 0.9 m, Oct 7-8, 30-min steps', 'NaCl 0.9 m, Nov 16',
    # 0.70 mol/kg KCl standard
    'KCl 8 S/m std, Nov 5', 'KCl 8 S/m std, Nov 6', 'KCl 8 S/m std, Nov 9',
    'KCl 8 S/m std, Nov 10', 'KCl 8 S/m std, Nov 11', 'KCl 8 S/m std, Nov 11-12',
]
ANCHOR = True
# Relative 1-bar reference uncertainty by source (ASSUMPTIONS for PI review)
ANCHOR_UNC = {'McCleskey (2011)': 0.01, 'McCleskey et al. (2012)': 0.02}
# Guide polynomial degree by number of points in the isotherm
GUIDE_DEGREE = [(9, 3), (4, 2), (0, 1)]   # (min points, degree)
# Panels: title, solute, molality, x-limit (MPa)
PANELS = [('NaCl 0.5 mol/kg', 'NaCl', 0.5, 900), ('NaCl 0.9 mol/kg', 'NaCl', 0.9, 290),
          ('KCl 0.70 mol/kg', 'KCl', 0.70, 290)]
ANCHOR_NOTE = ('1-bar values anchored to McCleskey (2011)/(2012); '
               'measured pressure dependence')

# Temperature palette shared with the supplement (validated colours; the
# 8/13 C and 16 C breaks are not, see plot_nacl_sigma_P.TEMP_COLORS)
TEMP_COLORS = [(8, '#184f95'), (13, '#5598e7'), (17.5, '#0f9fb0'), (21.5, '#1baf7a'),
               (24, '#eda100'), (27, '#e87ba4'), (99, '#a8322f')]
PAPER = '#fcfcfb'
MARK = {'Cortes 2025': 'o', 'Diaz 2026': 's'}


def temp_color(T_C):
    return next(c for t, c in TEMP_COLORS if T_C < t)


def load_main():
    """Kept steps of MAIN_RUNS, anchored if ANCHOR. Returns (data, factor table)."""
    allsteps = pd.concat([src.our_steps_2025(False), src.our_steps_2026(False)],
                         ignore_index=True)
    runs, rows = [], []
    for run in MAIN_RUNS:
        g = allsteps[allsteps.run == run]
        if g.empty:
            raise SystemExit(f'run not found in the step CSVs: {run}')
        info = src.anchor_info(g)
        rows.append(info)
        if ANCHOR:
            _, g = src.anchor_to_1bar(g)
        else:
            g = g.assign(anchor_source='McCleskey (2011)')  # label only
        g = g.assign(unc=g.anchor_source.map(ANCHOR_UNC) if ANCHOR else 0.0)
        runs.append(g)
    return pd.concat(runs, ignore_index=True), pd.DataFrame(rows)


def guide(g):
    """Polynomial guide through an isotherm, within the measured P range only."""
    n = len(g)
    deg = next(d for k, d in GUIDE_DEGREE if n >= k)
    c = np.polyfit(g.P_MPa, g.sigma_Sm, deg)
    P = np.linspace(g.P_MPa.min(), g.P_MPa.max(), 200)
    return P, np.polyval(c, P)


def draw_panel(ax, d, solute, m, xmax, title):
    items = []
    sub = d[(d.solute == solute) & (abs(d.molality - m) < 0.05)]
    for run, g in sub.groupby('run', sort=False):
        col = temp_color(g.T_C.mean())
        P, S = guide(g)
        ax.plot(P, S, '-', color=col, lw=1.1, zorder=2)
        lo = np.sqrt((g.sigma_Sm - g.sigma_lo) ** 2 + (g.unc * g.sigma_Sm) ** 2)
        hi = np.sqrt((g.sigma_hi - g.sigma_Sm) ** 2 + (g.unc * g.sigma_Sm) ** 2)
        for desc in (True, False):
            q = g.descent.to_numpy() == desc
            if q.any():
                ax.errorbar(g.P_MPa[q], g.sigma_Sm[q], yerr=[lo[q], hi[q]], fmt=MARK[g.source.iloc[0]],
                            color=col, mfc=PAPER if desc else col, mec=col if desc else PAPER,
                            ms=4.2, mew=0.8 if not desc else 1.0, elinewidth=0.7, capsize=0,
                            zorder=3)
        items.append((g.P_MPa.max(), S[-1], f'{g.T_C.mean():.1f}', col))
    if solute == 'NaCl' and m == 0.5:
        ah = src.adams_hall_nacl()
        up = ah[~ah.descent]
        ax.plot(up.P_MPa, up.sigma_Sm, '-', color='k', lw=0.8, zorder=2)
        ax.plot(ah.P_MPa, ah.sigma_Sm, 'P', color='k', ms=3.5, mew=0, zorder=3)
        ax.text(0.97, 0.97, r'black +: Adams \& Hall (1931)' '\n' r'25.00 $^\circ$C', transform=ax.transAxes,
                ha='right', va='top', fontsize=6.2)
    return items


def label_ends(ax, items, gap_frac=0.055):
    """Temperature labels just right of each curve end, spread vertically."""
    ylo, yhi = ax.get_ylim()
    gap = gap_frac * (yhi - ylo)
    order = np.argsort([it[1] for it in items])
    yt = np.array([it[1] for it in items], dtype=float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        yt[i] = max(yt[i], yt[j] + gap)
    x0, x1 = ax.get_xlim()
    for (x, y, text, col), y2 in zip(items, yt):
        ax.text(x + 0.02 * (x1 - x0), y2, text + r'$^\circ$C', color=col, fontsize=6.4,
                va='center', ha='left', clip_on=False)


def main():
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 8,
                         'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b'})
    d, table = load_main()
    table = table.rename(columns={'molality': 'm_mol_kg', 'sigma_meas_1bar': 'sigma_meas_1bar_Sm',
                                  'sigma_ref': 'sigma_ref_Sm'})
    table.round(5).to_csv(os.path.join(OUT, 'sigma_P_main_anchor_factors.csv'), index=False)
    print(table.round(4).to_string(index=False))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={'wspace': 0.42})
    for ax, (title, solute, m, xmax) in zip(axs, PANELS):
        sub = d[(d.solute == solute) & (abs(d.molality - m) < 0.05)]
        lo, hi = (sub.sigma_lo.min(), sub.sigma_hi.max())
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.02 * xmax, xmax)
        ax.set_xticks(np.arange(0, 701, 200) if xmax > 500 else np.arange(0, 201, 50))
        items = draw_panel(ax, d, solute, m, xmax, title)
        label_ends(ax, items)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('Pressure (MPa)')
        ax.grid(True, color='#e4e3df', lw=0.5)
        ax.set_axisbelow(True)
    axs[0].set_ylabel(r'$\sigma$ (S/m)')
    fig.text(0.5, 0.005, ANCHOR_NOTE + '.\n' r'Open symbols: decompression. Bars: step range and anchor '
             r'uncertainty (1\,\% McCleskey 2011, 2\,\% MC12) in quadrature.',
             ha='center', va='bottom', fontsize=6.2, color='#52514e')
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.25)
    for ext, kw in (('png', dict(dpi=250)), ('pdf', {})):
        fig.savefig(os.path.join(OUT, f'sigma_P_main.{ext}'), **kw)


if __name__ == '__main__':
    main()
