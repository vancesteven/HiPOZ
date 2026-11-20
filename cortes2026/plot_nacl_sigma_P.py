#!/usr/bin/env python3
"""
Conductivity-pressure data processing and the SUPPLEMENT figure sigma(P).

Pipeline role (details: nacl_sigma_P/README.md)
    1. This script fits every 2026 sweep (slow, ~30 s) and groups sweeps into
       pressure steps. It writes the step CSVs that all other code reads.
    2. The figure here is the supplement figure: ALL our data, diagnostics
       included. The main-text figure is plot_sigma_P_main.py (fast, CSVs only).
    3. The data model and every literature source (Fisher and Fox 1979, Fisher
       1962, Adams and Hall 1931, Pan 2021, McCleskey 2011/2012 references) live
       in sigma_P_sources.py; this script only plots them.

Supplement figure layout: two columns, NaCl (left) and KCl (right).
    top row     absolute sigma (S/m); KCl on a log axis because the dilute
                literature KCl (0.01-0.3 S/m) and our 0.70 mol/kg standard
                (5-9.5 S/m) differ by 30x
    bottom row  sigma / sigma(first 1-bar step of the same run), the same
                series: shows the pressure dependence and hysteresis without
                the absolute calibration
Marks: colour = temperature (TEMP_COLORS); filled = compression, hollow =
decompression; black x = excluded step (reasons below); triangle / diamond at
the left of the KCl panel = 1-bar bottle table / McCleskey et al. (2012) at the
Nov KCl runs' temperatures (a check, not forced to either); dashed/thick lines
= Pan et al. (2021) model (thick inside its fitted P-T range, dashed where
extrapolated); labels sit at the ends of the curves.

Our data
- Aug 2025 Cortes 0.5 mol/kg NaCl isotherms (small vessel).
- 2026 Diaz runs (large vessel): 0.9 mol/kg (5 wt%) NaCl Oct 7-8 (30-min steps),
  Oct 9 (10-min steps) and Nov 16; the 8 S/m KCl standard Oct 14-15 and Nov 5-12.
  sigma = K / R. K sources: the nearest ambient 8 S/m standard (Oct 14 for the
  Oct NaCl runs, an estimate: measured five days later, after the leak repair;
  open question 2026-10-19), each Oct 14-15 run's own first triplet, the common
  Nov K (120.617, median of the settled 1-bar steps in vessel_1bar_check.py) for
  Nov 5-12, and for NaCl Nov 16 the first 1-bar step against McCleskey (2011)
  because its potentiostat connection broke after loading.
- Left out entirely: the Aug 18 - Sep 10 2026 KCl runs (stepped too fast;
  Steve 2026-11-08); Oct 6 NaCl (5-10 min steps, up/down mismatch in Diaz's
  notebook), which is still fitted into the sweep table for the supplement tables.

Steps and exclusions
Sweeps are grouped into pressure steps (consecutive sweeps < 4 min apart at the
same nominal P); each point is a step mean with its min-max range. Some steps
were taken before T and P had equilibrated (Steve, 2026-11-07); they are listed
in EXCLUDED (2025) and PROVISIONAL_2026 / the zAnalysis config exclusions (2026)
with the reason, and drawn as black x. Nothing is removed from the configs.

Usage:  python cortes2026/plot_nacl_sigma_P.py            (about 50 s)
        python cortes2026/plot_nacl_sigma_P.py --figures-only   (redraw from the CSVs, seconds)
Writes: nacl_sigma_P/nacl_0p5m_sigma_P_steps.csv   2025 step table (excluded column)
        nacl_sigma_P/sigma_P_2026_steps.csv        2026 kept-step table
        diaz/diaz_hp_sweep_fits.csv                every 2026 sweep, with exclusion reasons
        nacl_sigma_P/kcl_literature_ratio.csv      KCl literature, absolute and ratio
        nacl_sigma_P/sigma_P_supplement.{pdf,png}  the supplement figure
        nacl_sigma_P/nacl_0p5m_sigma_P.{pdf,png}   DEPRECATED copy of the above,
            kept so references in paper drafts do not break. The former
            nacl_0p5m_sigma_P_excluded.png is gone: its excluded steps are now
            black x in this figure (their times are in the CSVs above).
"""
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)
import cortes_data_processing as cdp  # noqa: E402
import sigma_P_sources as src  # noqa: E402
import vessel_1bar_check as vbc  # noqa: E402
from gamryTools import Solution  # noqa: E402

OUT = os.path.join(HERE, 'nacl_sigma_P')
MAX_GAP_MPA = 150
# Isotherms whose top step joins the next lower pressure across a wider gap
# (Steve, 2026-11-08): the ~606 MPa, 20 C point to the ~400 MPa step. Aug 13
# 16:05 and Aug 19 16:35 both read 4.47 S/m there and overlap; the ~400 MPa
# step is Aug 13's, so the segment is Aug 13's.
ALWAYS_CONNECT = {('20250813', 20)}
SIGMA_STD = 8.0
DIRS = {'20250813Cortes': '20250813', '20250814Cortes': '20250814',
        '20250818Cortes': '20250818', '20250819Cortes': '20250819',
        '20250820Cortes': '20250820'}

# First sweep time (date_HHMMSS) of each excluded step -> reason
EXCLUDED = {
    '20250813_131431': 'outside HP vessel (bench check before loading)',
    '20250813_133324': 'outside HP vessel (bench check before loading)',
    '20250813_135054': 'outside HP vessel (bench check before loading)',
    '20250813_145441': 'labelled 450 MPa, gauge 613 MPa; -3.4% drift in triplet',
    '20250814_101348': 'overnight hold at 600 MPa; 14% below the previous afternoon',
    '20250814_150829': 'ambient bench series with the 0.75 and 2 mol/kg solutions',
    '20250818_150222': 'first 0 MPa triplet; +3.8% drift, settled 20 min later',
    '20250819_160357': 'bath still moving 18.9 -> 19.9 C; superseded at 16:35',
    '20250819_163922': 'hasty 20 C decompression (~2 min/step, +2-4% drift in triplet)',
    '20250819_164303': 'hasty 20 C decompression (~2 min/step, +2-4% drift in triplet)',
    '20250819_164715': 'hasty 20 C decompression (~2 min/step, +2-4% drift in triplet)',
    '20250819_165100': 'hasty 20 C decompression (~2 min/step, +2-4% drift in triplet)',
}

# Runs keyed by (date, bath setpoint to the nearest 5 C): colour, marker.
# Colours validated (dataviz validate_palette, light surface), cold -> warm;
# the three ~20 C days share one hue and differ by marker.
RUNS = {
    ('20250819', 0): ('#184f95', 'o'),
    ('20250819', 10): ('#5598e7', 'o'),
    ('20250818', 20): ('#1baf7a', 'p'),
    ('20250820', 20): ('#1baf7a', '^'),
    ('20250813', 20): ('#1baf7a', 'o'),
    ('20250820', 35): ('#a8322f', 'o'),
}

# 2026 runs: label, folders (fitted from the raw sweeps), solute, marker,
# K source. Colour comes from temperature (TEMP_COLORS), as for Aug 2025.
RUNS_2026 = [
    ('NaCl 0.9 m, Oct 7-8, 30-min steps', ['20261007', '20261008'], 'NaCl', 'D', 'oct14'),
    ('NaCl 0.9 m, Oct 9, 10-min steps', ['20261009'], 'NaCl', 'v', 'oct14'),
    ('KCl 8 S/m std, Oct 14-15', ['20261014', '20261015'], 'KCl', 's', 'self'),
    ('KCl 8 S/m std, Nov 5', ['20261105'], 'KCl', 's', 'nov_common'),
    ('KCl 8 S/m std, Nov 6', ['20261106'], 'KCl', 's', 'nov_common'),
    ('KCl 8 S/m std, Nov 9', ['20261109'], 'KCl', 's', 'nov_common'),
    ('KCl 8 S/m std, Nov 10', ['20261110'], 'KCl', 's', 'nov_common'),
    ('KCl 8 S/m std, Nov 11', ['20261111'], 'KCl', 's', 'nov_common'),
    ('KCl 8 S/m std, Nov 11-12', ['20261111', '20261112'], 'KCl', 's', 'nov_common'),
    ('NaCl 0.9 m, Nov 16', ['20261116'], 'NaCl', 'o', 'prior'),
]
# Folder 20261111 holds two runs, split by time: the 6.3 C morning run and the
# 29.5 C run that starts at 15:27 and continues into Nov 12.
RUN_WINDOWS = {
    'KCl 8 S/m std, Nov 11': (None, datetime(2026, 11, 11, 14, 0)),
    'KCl 8 S/m std, Nov 11-12': (datetime(2026, 11, 11, 14, 0), None),
}
# 'nov_common': the single K of all Nov 5-12 KCl runs, the median of the
# settled 1-bar steps (vessel_1bar_check.kcl_check). Their 1-bar points are a
# check against the bottle table and McCleskey (2012), not forced to either.
# 'prior': K from the run's first 1-bar step against McCleskey (2011) NaCl at
# the sample's composition (vessel_1bar_check.py). Nov 16: a soldered
# potentiostat connection broke after loading, so no standard applies
# (Steve, 2026-11-18).
# Provisional exclusions of 2026 sweeps (filename stem -> reason); the zAnalysis
# configs are not edited. Applied in load_2026 like config exclusions.
_PROV = 'provisional: +/-12 ohm CPE fits (~93 %), Steve 2026-11-18'
PROVISIONAL_2026 = {f'Default_20261112_{t}': _PROV for t in ('111506', '111559', '111652')}
PRIOR_MOLAL = {'NaCl 0.9 m, Nov 16': 0.902}
# Temperature -> colour, validated together (dataviz validate_palette,
# --pairs all, light surface): 1, 10, 19-20, 23, 25, 35 C. The 16 C cyan and
# the 8/13 C breaks (Nov 2026 KCl runs at 6, 11, 16 C) were NOT validated
# (validator not found).
TEMP_COLORS = [(8, '#184f95'), (13, '#5598e7'), (17.5, '#0f9fb0'), (21.5, '#1baf7a'),
               (24, '#eda100'), (27, '#e87ba4'), (99, '#a8322f')]


def temp_color(T_C):
    return next(c for t, c in TEMP_COLORS if T_C < t)
F_RANGE_HZ = 1e3 * np.array([10, 100])
# Fitted for the supplement tables only, not plotted: label, folders, solute,
# K source, reason
EXTRA_2026 = [
    ('NaCl 0.9 m, Oct 6, 5-10-min steps', ['20261006'], 'NaCl', 'oct14',
     'left out: 5-10 min steps, up/down mismatch (Diaz notebook p. 22-23)'),
]


def load_sweeps():
    frames = []
    for d, date in DIRS.items():
        df = cdp.load_cortes_data([d])
        df = df[(df.comp == 'NaCl') & (pd.to_numeric(df.w_molal, errors='coerce') == 0.5)].copy()
        hhmmss = df.filename.str.extract(r'(?:^|_)(\d{6})_')[0]
        df['dt'] = [datetime.strptime(date + h, '%Y%m%d%H%M%S') for h in hhmmss]
        frames.append(df)
    df = pd.concat(frames).sort_values('dt').reset_index(drop=True)
    df['T_C'] = df.T_K - 273.15
    new_step = (df.dt.diff().dt.total_seconds() > 240) | (df.P_MPa.diff().abs() > 25)
    df['step'] = new_step.cumsum()
    return df


def steps_table(df):
    rows = []
    for _, s in df.groupby('step'):
        key = s.dt.iloc[0].strftime('%Y%m%d_%H%M%S')
        sig = s.conductivity_Sm
        rows.append(dict(
            start=s.dt.iloc[0], key=key, n=len(s),
            P_MPa=s.P_MPa.mean(), T_C=s.T_C.mean(),
            sigma_Sm=sig.mean(), sigma_min=sig.min(), sigma_max=sig.max(),
            drift_pct=100 * (sig.iloc[-1] - sig.iloc[0]) / sig.iloc[0],
            excluded=EXCLUDED.get(key, '')))
    st = pd.DataFrame(rows)
    st['run'] = [(f'{t:%Y%m%d}', int(5 * round(T / 5))) for t, T in zip(st.start, st.T_C)]
    # The Aug 19 16:35 600 MPa triplet (equilibrated, 19.9 C) and the Aug 20
    # morning steps share the 20 C setpoint and the loading: one isotherm
    st['run'] = [('20250820', 20) if r == ('20250819', 20) else r for r in st.run]
    return st


def _excluded_files(folder):
    """Files marked exclude in a folder's config, except Oct 9, which is
    excluded only for its unresolved cell constant (cancels here)."""
    if folder == '20261009':
        return set()
    out = set()
    for cfg in glob(os.path.join(REPO, 'data', folder, 'zAnalysis*.csv')):
        c = pd.read_csv(cfg, dtype=str, keep_default_na=False)
        if 'exclude' in c:
            out |= set(c.loc[c.exclude.str.lower().isin(['x', 'true', '1']), 'filename'])
    return out


def fit_sweeps(folders):
    """CPE fit of every sweep in the folders; config exclusions flagged."""
    sw = []
    for folder in folders:
        skip = _excluded_files(folder)
        for f in sorted(glob(os.path.join(REPO, 'data', folder, 'Conductivity*', '*.txt'))):
            s = Solution(cmap_name='viridis')
            s.load_file(f)
            s.fit_circuit(circ_type='CPE', initial_guess=None, print_circuit=False,
                          basin_hopping=False, multiproc=False, f_range_hz=F_RANGE_HZ)
            sw.append(dict(time=s.time, filename=os.path.basename(f), folder=folder,
                           P_MPa=s.P_MPa, T_K=s.T_K, R=s.Rcalc_ohm, R_unc=s.Runc_ohm,
                           excluded='excluded in zAnalysis config'
                           if os.path.basename(f) in skip else
                           next((r for k, r in PROVISIONAL_2026.items()
                                 if os.path.basename(f).startswith(k)), '')))
    return pd.DataFrame(sw).sort_values('time').reset_index(drop=True)


def load_2026():
    """Step means of sigma = K / R for each 2026 run."""
    rows, sweeps, ex_rows = [], [], []
    for label, folders, comp, _, _ in RUNS_2026:
        sw_all_run = fit_sweeps(folders)
        t0, t1 = RUN_WINDOWS.get(label, (None, None))
        if t0 is not None:
            sw_all_run = sw_all_run[sw_all_run.time >= t0]
        if t1 is not None:
            sw_all_run = sw_all_run[sw_all_run.time < t1]
        sw_all_run = sw_all_run.reset_index(drop=True)
        exs = sw_all_run[sw_all_run.excluded != ''].reset_index(drop=True)
        if len(exs):
            e_new = (exs.time.diff().dt.total_seconds() > 240) | (exs.P_MPa.diff().abs() > 10)
            exs['step'] = e_new.cumsum()
            ex_rows.append(exs.groupby('step').agg(
                time=('time', 'first'), P_MPa=('P_MPa', 'mean'), T_K=('T_K', 'mean'),
                R=('R', 'mean'), reason=('excluded', 'first')).assign(run=label))
        sweeps.append(sw_all_run.assign(run=label, comp=comp))
        sw = sw_all_run[sw_all_run.excluded == ''].reset_index(drop=True)
        new = (sw.time.diff().dt.total_seconds() > 240) | (sw.P_MPa.diff().abs() > 10)
        sw['step'] = new.cumsum()
        st = sw.groupby('step').agg(time=('time', 'first'), P_MPa=('P_MPa', 'mean'),
                                    T_K=('T_K', 'mean'), R=('R', 'mean'),
                                    R_min=('R', 'min'), R_max=('R', 'max'), n=('R', 'size'))
        st['R_amb_first'] = sw.R.iloc[:3].mean()  # every run starts at ambient
        st['descent'] = st.P_MPa.cummax() > st.P_MPa + 5
        st['run'], st['comp'] = label, comp
        rows.append(st)
    for label, folders, comp, _, reason in EXTRA_2026:
        sweeps.append(fit_sweeps(folders).assign(run=label, comp=comp, excluded=reason))
    rec = pd.concat(rows, ignore_index=True)
    sw_all = pd.concat(sweeps, ignore_index=True)
    K = {lbl: SIGMA_STD * rec.loc[rec.run == lbl, 'R_amb_first'].iloc[0]
         for lbl, _, comp, _, _ in RUNS_2026 if comp == 'KCl'}
    K['oct14'] = K['KCl 8 S/m std, Oct 14-15']
    K['nov_common'] = vbc.kcl_check(vbc.ambient_steps(vbc.load_sweeps()))[1]
    for lbl, m in PRIOR_MOLAL.items():
        r = rec[rec.run == lbl]
        K[lbl] = vbc.sigma_prior_nacl(m, r.T_K.iloc[0] - 273.15) * r.R_amb_first.iloc[0]
    for lbl, _, _, _, ksrc in RUNS_2026 + [(e[0], None, None, None, e[3]) for e in EXTRA_2026]:
        if lbl not in set(rec.run):
            ms = sw_all.run == lbl
            sw_all.loc[ms, 'K_cell'] = K[ksrc]
            sw_all.loc[ms, 'sigma_Sm'] = K[ksrc] / sw_all.loc[ms, 'R']
            continue
        k = K[lbl if ksrc in ('self', 'prior') else ksrc]
        m = rec.run == lbl
        rec.loc[m, 'K_cell'] = k
        rec.loc[m, 'sigma_Sm'] = k / rec.loc[m, 'R']
        rec.loc[m, 'sigma_lo'] = k / rec.loc[m, 'R_max']
        rec.loc[m, 'sigma_hi'] = k / rec.loc[m, 'R_min']
        ms = sw_all.run == lbl
        sw_all.loc[ms, 'K_cell'] = k
        sw_all.loc[ms, 'sigma_Sm'] = k / sw_all.loc[ms, 'R']
    rec['T_C'] = rec.T_K - 273.15
    ex = pd.concat(ex_rows, ignore_index=True) if ex_rows else pd.DataFrame()
    if len(ex):
        ksrc = {lbl: k for lbl, _, _, _, k in RUNS_2026}
        ex['K_cell'] = [K[lbl if ksrc[lbl] in ('self', 'prior') else ksrc[lbl]] for lbl in ex.run]
        ex['sigma_Sm'] = ex.K_cell / ex.R
        ex['T_C'] = ex.T_K - 273.15
    sw_all.to_csv(os.path.join(HERE, 'diaz', 'diaz_hp_sweep_fits.csv'), index=False,
                  float_format='%.6g')
    return rec, ex


def prior_ticks():
    """Bottle-table and McCleskey (2012) sigma at each Nov KCl run's 1-bar steps."""
    k, _ = vbc.kcl_check(vbc.ambient_steps(vbc.load_sweeps()))
    out = {}
    for label, folders, comp, _, ksrc in RUNS_2026:
        if ksrc != 'nov_common':
            continue
        t0, t1 = RUN_WINDOWS.get(label, (None, None))
        m = k.folder.isin(folders)
        if t0 is not None:
            m &= k.start >= t0
        if t1 is not None:
            m &= k.start < t1
        out[label] = k[m]
    return out



# ------------------------------------------------------------------ plotting
INK, GREY, PAPER = '#0b0b0b', '#6b6b68', '#fcfcfb'
DEG = r'$^\circ$C'
MARKERS = {f"NaCl 0.5 m, Aug {int(d[6:])}, {s} C": mk for (d, s), (_, mk) in RUNS.items()}
MARKERS.update({lbl: mk for lbl, _, _, mk, _ in RUNS_2026})


def _style():
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
        'axes.edgecolor': '#52514e', 'axes.labelcolor': INK,
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
    })


def _spread(ys, gap):
    """Shift sorted label positions apart by at least `gap`, keeping order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        out[i] = max(out[i], out[j] + gap)
    return out


def label_ends(ax, items, x_text, gap, log=False):
    """Direct labels: text at x_text, y spread apart, a thin leader to (x, y).

    items: list of (x_end, y_end, text, colour[, linestyle]).  With log=True the
    spreading is done in decades (gap in decades), for a log y axis."""
    if not items:
        return
    f = np.log10 if log else (lambda v: v)
    g = (lambda v: 10 ** v) if log else (lambda v: v)
    xr = np.diff(ax.get_xlim())[0]
    ys = _spread([f(it[1]) for it in items], gap)
    for (x, y, text, col, *ls), yt in zip(items, ys):
        yt = g(yt)
        ax.plot([x, x_text - 0.016 * xr], [y, yt], ':', color='#9a9a9a', lw=0.7, zorder=1)
        ax.plot([x_text - 0.015 * xr, x_text - 0.003 * xr], [yt, yt], ls[0] if ls else '-',
                color=col, lw=2, zorder=5, clip_on=False)
        ax.text(x_text, yt, text, fontsize=7.4, color=INK, va='center', ha='left', zorder=5)


def prior_ticks():
    """Bottle-table and McCleskey (2012) sigma at each Nov KCl run's 1-bar steps."""
    k, _ = vbc.kcl_check(vbc.ambient_steps(vbc.load_sweeps()))
    out = {}
    for label, folders, comp, _, ksrc in RUNS_2026:
        if ksrc != 'nov_common':
            continue
        t0, t1 = RUN_WINDOWS.get(label, (None, None))
        m = k.folder.isin(folders)
        if t0 is not None:
            m &= k.start >= t0
        if t1 is not None:
            m &= k.start < t1
        out[label] = k[m]
    return out


def _marks(ax, x, y, mk, col, desc, ms=6, mew=1.2, yerr=None):
    """Filled markers for compression, hollow for decompression."""
    for d in (True, False):  # hollow first and larger: filled ones stay visible on top
        m = np.asarray(desc) == d
        if not m.any():
            continue
        e = None if yerr is None else [yerr[0][m], yerr[1][m]]
        ax.errorbar(np.asarray(x)[m], np.asarray(y)[m], yerr=e, fmt=mk, color=col,
                    mfc=PAPER if d else col, mec=col if d else PAPER,
                    ms=ms * (1.35 if d else 1), mew=mew if not d else 1.5,
                    elinewidth=1, capsize=0, zorder=3 if not d else 2.8)


def _plot_ours(ax, ours, solute, ycol, label_items, errbars=True):
    """Every run of one solute: kept steps (line + markers) and excluded steps (x)."""
    for run, g in ours[ours.solute == solute].groupby('run', sort=False):
        kept = g[g.excluded == '']
        if kept.empty:
            continue
        col, mk = temp_color(kept.T_C.mean()), MARKERS[run]
        if kept.source.iloc[0] == 'Cortes 2025':
            kept = kept.sort_values('P_MPa')
            P, S = kept.P_MPa.to_numpy(), kept[ycol].to_numpy()
            key = next(k for k in RUNS if run == f"NaCl 0.5 m, Aug {int(k[0][6:])}, {k[1]} C")
            for i in range(len(P) - 1):  # connect neighbouring steps only
                if P[i + 1] - P[i] <= MAX_GAP_MPA or (key in ALWAYS_CONNECT and i == len(P) - 2):
                    ax.plot(P[i:i + 2], S[i:i + 2], '-', color=col, lw=1.5, zorder=2)
        else:
            ax.plot(kept.P_MPa, kept[ycol], '-', color=col, lw=1.2, zorder=2)
        ye = ([kept.sigma_Sm - kept.sigma_lo, kept.sigma_hi - kept.sigma_Sm]
              if errbars else None)
        ye = None if ye is None else [np.asarray(v) for v in ye]
        _marks(ax, kept.P_MPa, kept[ycol], mk, col, kept.descent, yerr=ye)
        top = kept.loc[kept.P_MPa.idxmax()]
        conc = {'Cortes 2025': '0.5 m', 'Diaz 2026': f'{top.molality:.2f} m'}[top.source]
        label_items.append((top.P_MPa, top[ycol], f'{conc}, {top.run_short}, {top.T_C:.1f} {DEG}', col))
    ex = ours[(ours.solute == solute) & (ours.excluded != '') & ours[ycol].notna()]
    ax.plot(ex.P_MPa, ex[ycol], 'x', color=INK, ms=7, mew=1.5, zorder=4)


def _plot_adams(ax, df, ycol, items=None, text=None):
    up = df[~df.descent].sort_values('P_MPa')
    ax.plot(up.P_MPa, up[ycol], '-', color=INK, lw=0.9, zorder=2)
    _marks(ax, df.P_MPa, df[ycol], 'P', INK, df.descent, ms=6, mew=1.0)
    if items is not None:
        items.append((up.P_MPa.iloc[-1], up[ycol].iloc[-1], text, INK))


def _plot_pan(ax, ours, ratio):
    """Pan et al. (2021) on each measured NaCl isotherm (0.5 mol/kg: Aug 2025;
    0.9 mol/kg: 2026 NaCl runs). Thick = inside the model's fitted P-T range."""
    kept = ours[(ours.excluded == '') & (ours.solute == 'NaCl')]
    iso, seen = [], set()
    for (src_, run), g in kept.groupby(['source', 'run'], sort=False):
        m, T = round(g.molality.iloc[0], 1), g.T_C.mean()
        if 18 < T < 19.5 or (m, round(T)) in seen:  # one curve for the ~20 C days
            continue
        seen.add((m, round(T)))
        iso.append((m, T))
    for m, T in iso:
        P = np.linspace(0.1, 700 if m == 0.5 else 215, 300)
        p = src.pan_nacl(m, T, P)
        y = p.sigma_Sm / p.sigma_Sm.iloc[0] if ratio else p.sigma_Sm
        col = temp_color(T)
        ax.plot(p.P_MPa, y, '--', color=col, lw=1, alpha=0.75, zorder=1)
        ax.plot(np.where(p.in_envelope, p.P_MPa, np.nan), np.where(p.in_envelope, y, np.nan),
                '-', color=col, lw=2.6, alpha=0.35, zorder=1)


def _plot_kcl_lit(ax, lit, ycol, items, log):
    """Fisher and Fox (1979), Fisher (1962): absolute (S/m) or ratio."""
    for key, ls, col, lw in (('ff02', '-', INK, 1.1), ('ff002', '--', GREY, 1.0),
                             ('ff0005', ':', GREY, 1.0)):
        d = lit[key]
        ax.plot(d.P_MPa, d[ycol], ls, color=col, lw=lw, zorder=3,
                marker='^' if key == 'ff02' else None, ms=5)
        items.append((d.P_MPa.iloc[-1], d[ycol].iloc[-1],
                      f'KCl {d.molality.iloc[0]:g} M, Fisher \\& Fox 1979, 25 {DEG}', col, ls))
    for key, mfc in (('f62_01', GREY), ('f62_02', PAPER)):
        d = lit[key]
        ax.plot(d.P_MPa, d[ycol], 'v', color=GREY, mfc=mfc, ms=6, zorder=4)
        items.append((d.P_MPa.iloc[-1], d[ycol].iloc[-1],
                      f'KCl {d.molality.iloc[0]:g} M, Fisher 1962, 25 {DEG}', GREY))


def load_literature():
    """All literature series with 'ratio' added (sigma_P_sources.with_ratio)."""
    lit = dict(ff02=src.fisher_fox_kcl(0.02), ff002=src.fisher_fox_kcl(0.002),
               ff0005=src.fisher_fox_kcl(0.0005), f62_01=src.fisher_1962_kcl(0.01),
               f62_02=src.fisher_1962_kcl(0.02), ah_kcl=src.adams_hall_kcl(),
               ah_nacl=src.adams_hall_nacl())
    return {k: src.with_ratio(v) for k, v in lit.items()}


def kcl_literature_table(lit):
    """CSV of the KCl literature series (absolute and ratio), 25 C."""
    parts = [(lit[k].source.iloc[0], lit[k].molality.iloc[0], lit[k]) for k in
             ('ff02', 'ff002', 'ff0005', 'f62_01', 'f62_02', 'ah_kcl')]
    t = pd.concat([d[['P_MPa', 'ratio', 'sigma_Sm']].assign(source=s, c=c, T_C=src.LIT_T_C)
                   for s, c, d in parts], ignore_index=True)
    return t[['source', 'c', 'T_C', 'P_MPa', 'ratio', 'sigma_Sm']]


def build_figure(ours, lit):
    """Supplement figure: columns NaCl | KCl, rows absolute sigma | ratio to 1 bar."""
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.6), gridspec_kw={
        'height_ratios': [1, 0.55], 'hspace': 0.08, 'wspace': 0.2})
    (a_n, a_k), (r_n, r_k) = axs
    ticks = prior_ticks()
    # --- NaCl
    a_n.set_xlim(-22, 1000)
    r_n.set_xlim(-22, 1000)
    items = []
    _plot_pan(a_n, ours, ratio=False)
    _plot_pan(r_n, ours, ratio=True)
    _plot_ours(a_n, ours, 'NaCl', 'sigma_Sm', items)
    _plot_ours(r_n, ours, 'NaCl', 'ratio', [], errbars=False)
    ah = lit['ah_nacl']
    _plot_adams(a_n, ah, 'sigma_Sm', items,
                fr'0.505 m, Adams \& Hall 1931, 25 {DEG}')
    _plot_adams(r_n, ah, 'ratio')
    far = [e for e in items if e[0] > 400]
    near = [e for e in items if e[0] <= 400]
    label_ends(a_n, far, 712, gap=0.2)
    label_ends(a_n, near, 240, gap=0.2)
    a_n.set_title('NaCl', fontsize=12)
    # --- KCl
    a_k.set_yscale('log')
    a_k.set_ylim(0.005, 25)
    a_k.set_xlim(-25, 560)
    r_k.set_xlim(-25, 560)
    items = []
    _plot_ours(a_k, ours, 'KCl', 'sigma_Sm', items)
    _plot_ours(r_k, ours, 'KCl', 'ratio', [], errbars=False)
    for label, t in ticks.items():
        a_k.plot([-6] * len(t), t.sigma_prior_Sm, '>', color=INK, ms=4.5, mew=0, zorder=4, clip_on=False)
        a_k.plot([-13] * len(t), t.sigma_MC12_Sm, 'D', color='#8a8a86', ms=3.6, mew=0, zorder=4, clip_on=False)
    lit_items = []
    _plot_kcl_lit(a_k, lit, 'sigma_Sm', lit_items, log=True)
    _plot_kcl_lit(r_k, lit, 'ratio', [], log=False)
    kah = lit['ah_kcl']
    _plot_adams(a_k, kah, 'sigma_Sm', lit_items,
                fr'0.050 m, Adams \& Hall 1931, 25 {DEG}')
    _plot_adams(r_k, kah, 'ratio')
    label_ends(a_k, items, 245, gap=0.075, log=True)
    label_ends(a_k, lit_items, 292, gap=0.11, log=True)
    a_k.set_title('KCl', fontsize=12)
    # --- shared decoration
    a_n.set_ylabel(r'$\sigma$ (S/m)')
    r_n.set_ylabel(r'$\sigma(P)/\sigma(\mathrm{first\ 1\ bar\ step})$')
    r_n.set_ylim(0.88, 1.38)
    r_k.set_ylim(0.97, 1.16)
    for ax in (a_n, a_k, r_n, r_k):
        ax.grid(True, color='#e4e3df', lw=0.6)
        ax.set_axisbelow(True)
    for ax in (a_n, a_k):
        ax.tick_params(labelbottom=False)
    for ax in (r_n, r_k):
        ax.set_xlabel('Pressure (MPa)')
    r_n.set_xticks(np.arange(0, 701, 100))
    r_k.set_xticks(np.arange(0, 301, 50))
    a_k.set_xticks(np.arange(0, 301, 50))
    _style_key(fig)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.14)
    return fig


def _style_key(fig):
    """Key for line and marker styles only; curve identities are labelled in place."""
    from matplotlib.lines import Line2D
    g = '#52514e'
    h = [Line2D([], [], ls='', marker='o', color=g, ms=6),
         Line2D([], [], ls='', marker='o', color=g, mfc=PAPER, ms=6),
         Line2D([], [], ls='', marker='x', color=INK, ms=7, mew=1.5),
         Line2D([], [], ls='--', color=g, lw=1),
         Line2D([], [], ls='-', color=g, lw=2.6, alpha=0.35),
         Line2D([], [], ls='', marker='>', color=INK, ms=4.5),
         Line2D([], [], ls='', marker='D', color='#8a8a86', ms=3.6),
         Line2D([], [], ls='', marker='P', color=INK, ms=6)]
    fig.legend(h, ['compression (colour = temperature)', 'decompression',
                   'excluded step (reasons in the CSVs)', 'Pan et al. (2021), extrapolated',
                   'Pan et al. (2021), in its $P$--$T$ range',
                   'KCl 1 bar, bottle table at run $T$', 'KCl 1 bar, McCleskey et al. (2012)',
                   r'Adams \& Hall (1931), $\sigma_0$ from MC12'],
               fontsize=7.8, frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))


def tidy_ours():
    """Our steps (2025 + 2026, excluded included) with the ratio column."""
    return src.with_ratio(pd.concat([src.our_steps_2025(), src.our_steps_2026()],
                                    ignore_index=True))


def save_figures(ours, lit):
    fig = build_figure(ours, lit)
    for name in ('sigma_P_supplement', 'nacl_0p5m_sigma_P'):  # second: deprecated copy
        fig.savefig(os.path.join(OUT, name + '.png'), dpi=200)
        fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    _style()
    if '--figures-only' in sys.argv:  # reuse the CSVs from the last full run
        ours, lit = tidy_ours(), load_literature()
        save_figures(ours, lit)
        return
    st = steps_table(load_sweeps())
    missing = set(EXCLUDED) - set(st.key)
    if missing:
        raise SystemExit(f'EXCLUDED keys not found among steps: {sorted(missing)}')
    unassigned = set(st[st.excluded == ''].run) - set(RUNS)
    if unassigned:
        raise SystemExit(f'kept steps with no run style: {sorted(unassigned)}')
    st.drop(columns='key').round(4).to_csv(os.path.join(OUT, 'nacl_0p5m_sigma_P_steps.csv'),
                                           index=False)
    recent, ex26 = load_2026()
    recent.round(5).to_csv(os.path.join(OUT, 'sigma_P_2026_steps.csv'), index=False)
    # the figure reads the CSVs just written, through sigma_P_sources
    ours, lit = tidy_ours(), load_literature()
    kcl_literature_table(lit).round(5).to_csv(os.path.join(OUT, 'kcl_literature_ratio.csv'),
                                              index=False)
    save_figures(ours, lit)
    print(f'{(st.excluded == "").sum()} steps kept, {(st.excluded != "").sum()} excluded -> {OUT}')


if __name__ == '__main__':
    main()
