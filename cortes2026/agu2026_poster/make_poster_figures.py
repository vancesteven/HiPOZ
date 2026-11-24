#!/usr/bin/env python3
"""
AGU 2026 poster figures (A: sigma vs P, 3 panels; B: relative sigma vs P for 0.9 mol/kg
NaCl; C: percent increase to 200 MPa vs T).

Same data, run selection, exclusions and anchoring as plot_sigma_P_main.py (it is
imported; nothing is re-fitted). Numbers shown come from poster_facts.compute().

    echo n | python cortes2026/agu2026_poster/make_poster_figures.py

Writes fig_A_sigma_P.{png,pdf}, fig_B_relative_sigma_0p9m.{png,pdf},
fig_C_pct_increase_vs_T.{png,pdf} next to this file. Each figure is created at its
PRINT size (inches), so every font size in points below is the size on the poster.
PNG at 300 dpi, PDF vector. No LaTeX needed (mathtext).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import poster_facts as pf  # noqa: E402  (imports the main-figure module and sigma_P_sources)
pm, src = pf.pm, pf.src

# ---------------------------------------------------------------- style
FS_AXLAB, FS_TICK, FS_TITLE, FS_LABEL, FS_NOTE = 22, 18, 24, 18, 16   # points at print size
LW, MS, MEW, ELW = 3.0, 10.5, 2.4, 2.0
PAPER = 'white'
SIZES = {'A': (20.0, 7.5), 'B': (9.0, 7.5), 'C': (7.0, 6.0)}   # inches, print size
FLAG_10C = True   # annotate the 9.85 C Aug-2025 isotherm whose 1-bar step looks unsettled (facts h)

# Colour = temperature, continuous, cold = blue ... warm = red. NOT the main figure's
# TEMP_COLORS bins (those are not monotone: violet at 4-8 C, pink at 24-27 C).
T_ANCHORS = [(0, '#13307a'), (9, '#2a7fc4'), (16.5, '#4fb0d6'), (19, '#9c9c9c'), (21.5, '#f4b41a'), (29, '#d5461b'), (36, '#8c1515')]
CMAP = LinearSegmentedColormap.from_list(
    'temp', [((t - T_ANCHORS[0][0]) / (T_ANCHORS[-1][0] - T_ANCHORS[0][0]), c) for t, c in T_ANCHORS])


def tcolor(T):
    return CMAP((min(max(T, 0), 36) - 0) / 36.0)


def cvd_report(Ts):
    """Smallest CAM02-UCS colour difference between isotherms >= 4 K apart, for normal
    vision and simulated deutan/protan/tritan deficiency (severity 100)."""
    from colorspacious import cspace_convert, deltaE
    out = {}
    for name, sp in (('normal', None), ('deuteranopia', 'deuteranomaly'), ('protanopia', 'protanomaly'),
                     ('tritanopia', 'tritanomaly')):
        rgb = np.array([tcolor(T)[:3] for T in Ts])
        if sp:
            rgb = np.clip(cspace_convert(rgb, {'name': 'sRGB1+CVD', 'cvd_type': sp, 'severity': 100}, 'sRGB1'), 0, 1)
        m = np.inf
        for i, a in enumerate(Ts):
            for j, b in enumerate(Ts):
                if j > i and abs(a - b) >= 4:
                    m = min(m, float(deltaE(rgb[i], rgb[j], input_space='sRGB1')))
        out[name] = m
    return out


def setup():
    plt.rcParams.update({
        'text.usetex': False, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
        'font.size': FS_TICK, 'axes.labelsize': FS_AXLAB, 'axes.titlesize': FS_TITLE,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'axes.linewidth': 2.0,
        'xtick.major.width': 2.0, 'ytick.major.width': 2.0, 'xtick.major.size': 8, 'ytick.major.size': 8,
        'figure.facecolor': PAPER, 'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
        'axes.edgecolor': '#222222', 'axes.labelcolor': '#111111', 'text.color': '#111111',
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def deg(T, nd=1):
    return '%.*f\u00b0C' % (nd, T)


def place_labels(ax, items, x_off_frac=0.018, fs=FS_LABEL, gap_pts=None):
    """Direct labels right of each curve end. items = (x, y, text, colour). Labels are pushed
    apart vertically (in display space) so none overlap; a thin leader joins a displaced label
    to its curve end."""
    fig = ax.figure
    fig.canvas.draw()
    gap = (gap_pts or fs * 1.18) * fig.dpi / 72.0
    tr = ax.transData
    pts = np.array([tr.transform((x, y)) for x, y, _, _ in items])
    order = np.argsort(pts[:, 1])
    ypix = pts[:, 1].copy()
    for k in range(1, len(order)):
        ypix[order[k]] = max(ypix[order[k]], ypix[order[k - 1]] + gap)
    # re-centre the pushed group so labels do not all drift upward
    shift = 0.5 * (np.mean(ypix - pts[:, 1]))
    ypix -= shift
    for k in range(1, len(order)):
        ypix[order[k]] = max(ypix[order[k]], ypix[order[k - 1]] + gap)
    x0, x1 = ax.get_xlim()
    inv = tr.inverted()
    for (x, y, text, col), yp in zip(items, ypix):
        xt = x + x_off_frac * (x1 - x0) * 2.2
        yt = inv.transform((0, yp))[1]
        ax.text(xt, yt, text, color=col, fontsize=fs, va='center', ha='left', clip_on=False,
                fontweight='bold')
        if abs(yt - y) > 0.4 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * gap / ax.bbox.height:
            ax.plot([x + 0.004 * (x1 - x0), xt - 0.004 * (x1 - x0)], [y, yt], '-', color=col, lw=1.6,
                    clip_on=False, zorder=1)


def draw_points(ax, g, xcol, ycol, col, lo=None, hi=None, fill_desc=True):
    for desc in (True, False):
        q = g.descent.to_numpy() == desc
        if not q.any():
            continue
        kw = dict(fmt='o', ms=MS, mew=MEW, color=col, capsize=0, elinewidth=ELW, zorder=3,
                  mfc=PAPER if desc else col, mec=col if desc else PAPER)
        if lo is None:
            ax.plot(g[xcol][q], g[ycol][q], 'o', ms=MS, mew=MEW, mfc=PAPER if desc else col,
                    mec=col if desc else PAPER, zorder=3)
            ax.lines[-1].set_color(col)
        else:
            ax.errorbar(g[xcol][q], g[ycol][q], yerr=[lo[q], hi[q]], **kw)


# ---------------------------------------------------------------- Fig A
def fig_A(res):
    d = res['d']
    fig, axs = plt.subplots(1, 3, figsize=SIZES['A'])
    panels = [('NaCl 0.5 mol/kg, 0-700 MPa', 'NaCl', 0.5, (-18, 930), np.arange(0, 701, 200)),
              ('NaCl 0.9 mol/kg, 0-200 MPa', 'NaCl', 0.9, (-6, 300), np.arange(0, 201, 50)),
              ('KCl 0.70 mol/kg, 0-200 MPa', 'KCl', 0.70, (-6, 300), np.arange(0, 201, 50))]
    for ax, (title, sol, m, xl, xt) in zip(axs, panels):
        sub = d[(d.solute == sol) & (abs(d.molality - m) < 0.05)]
        ylo, yhi = sub.sigma_lo.min(), sub.sigma_hi.max()
        pad = 0.08 * (yhi - ylo)
        ax.set_ylim(ylo - pad, yhi + pad * (2.2 if m == 0.5 else 1))
        ax.set_xlim(*xl)
        ax.set_xticks(xt)
        items = []
        for run, g in sub.groupby('run', sort=False):
            col = tcolor(g.T_C.mean())
            P, S = pm.guide(g)
            ax.plot(P, S, '-', color=col, lw=LW, zorder=2, solid_capstyle='round')
            lo = np.sqrt((g.sigma_Sm - g.sigma_lo) ** 2 + (g.unc * g.sigma_Sm) ** 2)
            hi = np.sqrt((g.sigma_hi - g.sigma_Sm) ** 2 + (g.unc * g.sigma_Sm) ** 2)
            draw_points(ax, g, 'P_MPa', 'sigma_Sm', col, lo.to_numpy(), hi.to_numpy())
            items.append((g.P_MPa.max(), S[-1], deg(g.T_C.mean()), col))
        if m == 0.5:
            ah = res['ah']
            up, full = ah['up'], ah['ah']
            ax.plot(up.P_MPa, up.sigma_Sm, '-', color='k', lw=2.4, zorder=2.5)
            ax.plot(full.P_MPa, full.sigma_Sm, 'P', color='k', ms=11, mew=0, zorder=3.5)
            ax.annotate('Adams & Hall (1931),\n25.00\u00b0C (black +)', xy=(330, 4.74), xytext=(300, 5.1),
                        fontsize=FS_LABEL, color='k', ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='-', color='k', lw=1.8, shrinkA=2, shrinkB=2))
        place_labels(ax, items)
        ax.set_title(title, pad=14)
        ax.set_xlabel('Pressure (MPa)')
        ax.grid(True, color='#dcdcdc', lw=1.2)
        ax.set_axisbelow(True)
    axs[0].set_ylabel(r'Conductivity $\sigma$ (S/m)')
    fig.text(0.5, 0.012, 'Colour = temperature.  Filled = compression, open = decompression.  '
             '1-bar values anchored to McCleskey (2011, 2012).',
             ha='center', va='bottom', fontsize=FS_NOTE, color='#333333')
    fig.subplots_adjust(left=0.055, right=0.992, top=0.90, bottom=0.2, wspace=0.27)
    return fig


# ---------------------------------------------------------------- Fig B
def fig_B(res):
    d, ah = res['d'], res['ah']
    fig, ax = plt.subplots(figsize=SIZES['B'])
    sub = d[(d.solute == 'NaCl') & (abs(d.molality - 0.9) < 0.05)]
    items = []
    ax.axhline(1.0, color='#888888', lw=1.8, zorder=1)
    for run, g in sub.groupby('run', sort=False):
        g = g.sort_values('time')
        s1 = src.first_ambient(g).sigma_Sm
        g = g.assign(ratio=g.sigma_Sm / s1)
        col = tcolor(g.T_C.mean())
        c = np.polyfit(g.P_MPa, g.ratio, 3)
        P = np.linspace(g.P_MPa.min(), g.P_MPa.max(), 200)
        ax.plot(P, np.polyval(c, P), '-', color=col, lw=LW, zorder=2)
        draw_points(ax, g, 'P_MPa', 'ratio', col)
        items.append((P[-1], float(np.polyval(c, P[-1])), deg(g.T_C.mean()), col))
    # Adams and Hall, 25 C, 0.5 mol/kg
    up = ah['up']
    ax.plot(up.P_MPa[up.P_MPa < 260], up.sigma_Sm[up.P_MPa < 260] / up.sigma_Sm.iloc[0], '--', color='k', lw=3.0,
            dashes=(5, 2.5), zorder=2.5)
    upz = up[up.P_MPa < 260]
    ax.plot(upz.P_MPa, upz.sigma_Sm / up.sigma_Sm.iloc[0], 'P', color='k', ms=12, mew=0, zorder=3.5)
    # Pan et al. (2021), extrapolated, at two temperatures inside their T range
    Pg = np.linspace(0.1, 200, 100)
    for T in (1.7, 20.6):
        r = pf.pan_curve(0.902, T, Pg)
        ax.plot(Pg, r, ':', color=tcolor(T), lw=4.2, zorder=2.2, dash_capstyle='round')
    ax.set_xlim(-6, 292)
    ax.set_ylim(0.88, 1.155)
    ax.set_xticks(np.arange(0, 201, 50))
    place_labels(ax, items, fs=FS_LABEL - 1)
    ax.text(75, 0.987, 'Pan et al. (2021),\n20.6\u00b0C, extrapolated', color=tcolor(20.6), fontsize=FS_LABEL - 1,
            va='top', ha='left', fontweight='bold')
    ax.text(5, 0.918, 'Pan et al. (2021),\n1.7\u00b0C, extrapolated', color=tcolor(1.7), fontsize=FS_LABEL - 1,
            va='top', ha='left', fontweight='bold')
    ax.annotate('Adams & Hall (1931)\n0.5 mol/kg, 25\u00b0C', xy=(108, 1.0607), xytext=(2, 1.108), fontsize=FS_LABEL - 1,
                color='k', va='bottom', ha='left',
                arrowprops=dict(arrowstyle='-', color='k', lw=1.8, shrinkA=2, shrinkB=2))
    ax.set_title('NaCl 0.9 mol/kg, 0-200 MPa', pad=14)
    ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel(r'$\sigma(P)\,/\,\sigma$(1 bar)')
    ax.grid(True, color='#dcdcdc', lw=1.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.012, 'Filled = compression, open = decompression.  Pan et al. data: 212-1713 MPa.',
             ha='center', va='bottom', fontsize=FS_NOTE - 1.5, color='#333333')
    fig.subplots_adjust(left=0.15, right=0.985, top=0.91, bottom=0.17)
    return fig


# ---------------------------------------------------------------- Fig C
def fig_C(res):
    t, ah = res['tab'], res['ah']
    fig, ax = plt.subplots(figsize=SIZES['C'])
    spec = [('NaCl 0.5', 'D', '-', 'NaCl 0.5 mol/kg'), ('NaCl 0.9', 's', '--', 'NaCl 0.9 mol/kg'),
            ('KCl 0.70', '^', ':', 'KCl 0.70 mol/kg')]
    for sol, mk, ls, lab in spec:
        s = t[(t.solution == sol) & t.pct_200.notna()].sort_values('T_C')
        err = s.pct_200_steprange if sol == 'NaCl 0.5' else s.ud_max_pp
        ax.plot(s.T_C, s.pct_200, ls, color='#444444', lw=2.4, zorder=1)
        ax.errorbar(s.T_C, s.pct_200, yerr=err, fmt='none', ecolor='#222222', elinewidth=ELW, capsize=6, capthick=ELW, zorder=2)
        for r, e in zip(s.itertuples(), err):
            ax.plot(r.T_C, r.pct_200, mk, ms=MS + 1.5, mfc=tcolor(r.T_C), mec='#111111', mew=1.6, zorder=3)
    ax.plot([25], [ah['pct_200']], '*', ms=24, mfc='k', mec='k', zorder=4)
    ax.annotate('Adams & Hall (1931)\n0.5 mol/kg, 25\u00b0C', xy=(25, ah['pct_200']), xytext=(28.2, 0.4), fontsize=FS_LABEL - 4,
                ha='center', va='bottom', arrowprops=dict(arrowstyle='-', color='k', lw=1.8, shrinkA=2, shrinkB=14))
    ax.text(37.5, 17.0, 'NaCl 0.5 mol/kg', fontsize=FS_LABEL, ha='right', va='bottom', color='#222222', fontweight='bold')
    ax.text(-0.5, 2.4, 'NaCl 0.9 mol/kg', fontsize=FS_LABEL, ha='left', va='bottom', color='#222222', fontweight='bold')
    ax.text(15.0, 14.6, 'KCl\n0.70 mol/kg', fontsize=FS_LABEL, ha='right', va='bottom', color='#222222', fontweight='bold')
    if FLAG_10C:
        r = t[(t.solution == 'NaCl 0.5') & (abs(t.T_C - 9.85) < 0.3)].iloc[0]
        ax.annotate('1-bar step\nunsettled?', xy=(r.T_C, r.pct_200), xytext=(r.T_C + 3.0, r.pct_200 - 1.5), fontsize=FS_LABEL - 2,
                    ha='left', va='center', arrowprops=dict(arrowstyle='-', color='#333333', lw=1.5, shrinkA=2, shrinkB=8))
    ax.set_xlim(-1.5, 38)
    ax.set_ylim(0, 37)
    ax.set_xlabel('Temperature (\u00b0C)')
    ax.set_ylabel(r'$\sigma$ increase to 200 MPa (%)')
    ax.set_title('Increase to 200 MPa', pad=14)
    ax.grid(True, color='#dcdcdc', lw=1.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.012, 'Bars: |down - up| (NaCl 0.9, KCl);\nstep range (NaCl 0.5, no decompression kept).',
             ha='center', va='bottom', fontsize=FS_NOTE - 1.5, color='#333333')
    fig.subplots_adjust(left=0.15, right=0.975, top=0.91, bottom=0.25)
    return fig


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(HERE, f'{name}.{ext}'), dpi=300)
    plt.close(fig)


def main():
    setup()
    res = pf.compute()
    Ts = sorted(set(res['tab'].T_C.round(2)))
    print('CVD min CAM02-UCS dE between isotherm colours >= 4 K apart:', {k: round(v, 1) for k, v in cvd_report(Ts).items()})
    for key, fn, name in (('A', fig_A, 'fig_A_sigma_P'), ('B', fig_B, 'fig_B_relative_sigma_0p9m'),
                          ('C', fig_C, 'fig_C_pct_increase_vs_T')):
        fig = fn(res)
        save(fig, name)
        w, h = SIZES[key]
        print(f'{name}: print size {w} x {h} in -> {int(w * 300)} x {int(h * 300)} px at 300 dpi')


if __name__ == '__main__':
    main()
