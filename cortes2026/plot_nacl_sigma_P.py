#!/usr/bin/env python3
"""
Clean sigma(P) for the Gamry high-pressure cell, all on one axis.

- Aug 2025 Cortes 0.5 mol/kg NaCl isotherms (circles, triangles, squares by
  day; colour by temperature).
- 2026 Diaz runs with the correct trend (Steve, 2026-11-08): 0.9 mol/kg
  (5 wt%) NaCl Oct 7-8 (30-min steps) and Oct 9 (10-min steps); the 8 S/m
  KCl standard Oct 14-15, Nov 5 and Nov 6 (squares). sigma = K / R with the
  ambient-pressure 8 S/m standard of the nearest standard run setting K:
  Oct 14 for the Oct NaCl runs (an estimate: measured five days later, after
  the leak repair; open question 2026-10-19), each run's own first triplet
  for the standard runs. NaCl 0.9 m Nov 16 (circles): K from its first 1-bar
  step against McCleskey (2011), since its connection broke after loading.
- Pan et al. (2021) NaCl model (pan2021_nacl.py) at 0.5 and 0.9 mol/kg on
  the measured isotherms: solid inside the experimental envelope
  (212-1713 MPa, 233-295 K), dashed where extrapolated.
- Adams & Hall (1931) 2.8650 wt% (0.5047 mol/kg) NaCl at 25.00 C, Table I
  (cortes2026/literature/adams_hall_1931_nacl_2p865.csv). They report only
  R/R0, so sigma = sigma0 R0/R with sigma0 from McCleskey et al. (2012) at
  25 C and that molality (the anchor is a model value, said in the legend).
Lower panel: kappa(P)/kappa(0.1 MPa) for KCl. Ours: R_amb_first / R of each KCl
standard run (0.70 mol/kg). Literature (all 25 C, dilute KCl, 0.0005-0.05 mol/kg):
Fisher & Fox (1979) Table I, 0.02 mol/L, and Fisher (1962) Tables III/VIII
(cortes2026/literature/fisher_*_kcl_25C.csv); Adams & Hall (1931) KCl 0.373 wt%
(ThermoLit/data/adams_hall_1931), kappa_P/kappa_0 = 1/(R/R0). Molar conductances
convert to conductivity ratios through the solvent density: kappa = Lambda c and
c rises with the water density, so kappa_P/kappa_1 = (Lambda_P/Lambda_1) rho_r,
rho_r = rho_water(P, 25 C)/rho_water(1 atm) from SeaFreeze 'water1' (Fisher 1962:
Dorsey rho_r tabulated in the CSV). Table: nacl_sigma_P/kcl_literature_ratio.csv.
Left out: the Aug 18 - Sep 10 2026 KCl runs, stepped too fast for the trend
with pressure (Steve, 2026-11-08); Oct 6 NaCl (5-10 min steps, up/down
mismatch in Diaz's notebook).

Sweeps are grouped into pressure steps (consecutive sweeps < 4 min apart at
the same nominal P); each point is a step mean with its min-max range. Near
room temperature some steps were taken before T and P had equilibrated and
break the concave-down sigma(P) seen on the other days (Steve, 2026-11-07).
Those steps are listed in EXCLUDED with the reason; nothing is removed from
the zAnalysis configs.

Usage:  python cortes2026/plot_nacl_sigma_P.py
Writes: cortes2026/nacl_sigma_P/nacl_0p5m_sigma_P.{pdf,png}   clean figure
        cortes2026/nacl_sigma_P/sigma_P_2026_steps.csv         2026 step table
        cortes2026/diaz/diaz_hp_sweep_fits.csv                 2026 per-sweep fits
        cortes2026/nacl_sigma_P/nacl_0p5m_sigma_P_excluded.png shows what was cut
        cortes2026/nacl_sigma_P/nacl_0p5m_sigma_P_steps.csv    step table
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
import pan2021_nacl as pan  # noqa: E402
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
ADAMS_CSV = os.path.join(HERE, 'literature', 'adams_hall_1931_nacl_2p865.csv')
ADAMS_MOLAL = 0.02865 / (1 - 0.02865) / 0.05844
ADAMS_T_C = 25.0
# ThermoLit is a sibling checkout (../ThermoLit); override with THERMOLIT_DIR
THERMOLIT = os.environ.get('THERMOLIT_DIR', os.path.join(os.path.dirname(REPO), 'ThermoLit'))
ADAMS_KCL_CSV = os.path.join(THERMOLIT, 'data', 'adams_hall_1931',
                             'table1_kcl_0.3730wt_25.00C.csv')
FISHER_FOX_CSV = os.path.join(HERE, 'literature', 'fisher_fox_1979_kcl_25C.csv')
FISHER_1962_CSV = os.path.join(HERE, 'literature', 'fisher_1962_kcl_25C.csv')
ATM_MPA = 0.101325
LIT_T_C = 25.0
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


def _style():
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
        'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
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


def label_ends(ax, items, x_text, gap):
    """Direct labels: text at x_text, y spread apart, a thin leader to (x, y).

    items: list of (x_end, y_end, text, colour)."""
    if not items:
        return
    ys = _spread([it[1] for it in items], gap)
    for (x, y, text, col, *ls), yt in zip(items, ys):
        # Dotted grey leader from the curve end; a short swatch in the curve's
        # colour and line style next to the text (no marker, so it can't be
        # mistaken for a data point)
        ax.plot([x, x_text - 16], [y, yt], ':', color='#9a9a9a', lw=0.7, zorder=1)
        ax.plot([x_text - 15, x_text - 3], [yt, yt], ls[0] if ls else '-', color=col,
                lw=2, zorder=5, clip_on=False)
        ax.text(x_text, yt, text, fontsize=7.6, color='#0b0b0b', va='center', ha='left', zorder=5)


def plot(st, path, show_excluded=False, recent=None, ex26=None):
    two = recent is not None and not show_excluded
    if two:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 10.0), sharex=True,
                                      gridspec_kw={'height_ratios': [1, 0.4], 'hspace': 0.08})
    else:
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ends, pan_ends = [], []
    if show_excluded and recent is not None:
        _plot_recent(ax, recent)
        if ex26 is not None and len(ex26):
            ax.plot(ex26.P_MPa, ex26.sigma_Sm, 'x', color='#0b0b0b', ms=7, mew=1.5, zorder=4)
            for i, (_, r) in enumerate(ex26.sort_values('P_MPa').iterrows()):
                dx, dy = ((8, 9), (8, 9), (-8, -16), (8, 24))[i % 4]  # staggered
                ax.annotate(f"{r.time:%m/%d %H:%M}", (r.P_MPa, r.sigma_Sm),
                            textcoords='offset points', xytext=(dx, dy), ha='right' if dx < 0 else 'left', va='center',
                            fontsize=6.5, color='#52514e',
                            arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.5))
    if recent is not None and not show_excluded:
        pan_ends = _plot_pan(ax, st, recent)
    keep = st[st.excluded == '']
    for run, (col, mk) in RUNS.items():
        r = keep[keep.run.map(lambda x: x == run)].sort_values('P_MPa')
        if r.empty:
            continue
        label = f"{datetime.strptime(run[0], '%Y%m%d'):%b} {int(run[0][6:])}, {r.T_C.mean():.1f}" 
        yerr = [r.sigma_Sm - r.sigma_min, r.sigma_max - r.sigma_Sm]
        # Connect neighbouring steps only; leave unmeasured spans open
        P, S = r.P_MPa.to_numpy(), r.sigma_Sm.to_numpy()
        for i in range(len(P) - 1):
            # ALWAYS_CONNECT bridges only the gap up to its highest pressure
            if P[i + 1] - P[i] <= MAX_GAP_MPA or (run in ALWAYS_CONNECT and i == len(P) - 2):
                ax.plot(P[i:i + 2], S[i:i + 2], '-', color=col, lw=1.5, zorder=2)
        ax.errorbar(r.P_MPa, r.sigma_Sm, yerr=yerr, fmt=mk, color=col, ms=6.5,
                    mec='#fcfcfb', mew=1.2, elinewidth=1, capsize=0, zorder=3)
        ends.append((P[-1], S[-1], r'NaCl 0.5 m, ' + label + r' $^\circ$C', col))
    if show_excluded:
        ex = st[st.excluded != '']
        ax.plot(ex.P_MPa, ex.sigma_Sm, 'x', color='#0b0b0b', ms=7, mew=1.5,
                label='excluded', zorder=4)
        j = -1
        for _, r in ex.sort_values(['P_MPa', 'sigma_Sm']).iterrows():
            j += r.P_MPa <= 12
            ax.annotate(f"{r.start:%m/%d %H:%M}", (r.P_MPa, r.sigma_Sm),
                        textcoords='offset points',
                        xytext=(6, -2) if r.P_MPa > 12 else (18, (-44, -30, -16, -2, 12, 26)[j % 6]),
                        fontsize=6.5, color='#52514e', va='center',
                        arrowprops=None if r.P_MPa > 12 else dict(arrowstyle='-', color='#9a9a9a', lw=0.5))
    if not two:
        ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel(r'$\sigma$ (S/m)')
    ax.grid(True, color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)
    recent_ends = []
    if recent is not None and not show_excluded:
        ends += pan_ends
        recent_ends = _plot_recent(ax, recent)
        ends.append(_plot_adams(ax))
    ax.set_title(r'Gamry high-pressure cell: NaCl (Cortes 2025, Diaz 2026) and the 8 S/m KCl standard',
                 fontsize=11)
    if show_excluded:
        ax.set_xlim(-22, 720)
        ax.legend(fontsize=7.8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    else:
        # Curves ending above 400 MPa: labels to the right of 700 MPa. The
        # 2026 runs (ending near 200 MPa): labels in the open space beside them.
        far = [e for e in ends if e[0] > 400]
        near = [e for e in ends if e[0] <= 400] + recent_ends
        ax.set_xlim(-22, 960)
        ax.set_xticks(np.arange(0, 701, 100))
        label_ends(ax, far, 712, gap=0.16)
        label_ends(ax, near, 240, gap=0.16)
        _style_key(ax)
    if two:
        _plot_kcl_ratio(ax2, recent)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if path.endswith('.png') and not show_excluded:
        fig.savefig(path[:-4] + '.pdf')
    plt.close(fig)


def _style_key(ax):
    """Key for line and marker styles only; curve identities are labelled in place."""
    from matplotlib.lines import Line2D
    g = '#52514e'
    h = [Line2D([], [], ls='--', color=g, lw=1),
         Line2D([], [], ls='-', color=g, lw=2.6, alpha=0.35),
         Line2D([], [], ls='', marker='s', color=g, ms=6),
         Line2D([], [], ls='', marker='s', color=g, mfc='#fcfcfb', ms=6),
         Line2D([], [], ls='', marker='X', color=g, ms=7),
         Line2D([], [], ls='', marker='>', color='#0b0b0b', ms=4.5),
         Line2D([], [], ls='', marker='D', color='#8a8a86', ms=3.6)]
    ax.legend(h, ['Pan et al. (2021), extrapolated', 'Pan et al. (2021), in its P--T range',
                  'KCl standard (squares)', 'hollow: decompression',
                  r'Adams \& Hall (1931), $\sigma_0$ from MC12',
                  'Nov KCl 1 bar, bottle table at run $T$',
                  'Nov KCl 1 bar, McCleskey (2012) at run $T$'],
              fontsize=7.4, frameon=False, loc='upper right')


def _plot_recent(ax, rec):
    ends = []
    ticks = prior_ticks()
    for label, _, comp, mk, _ in RUNS_2026:
        r = rec[rec.run == label]
        col = temp_color(r.T_C.mean())
        ax.plot(r.P_MPa, r.sigma_Sm, '-', color=col, lw=1.2, zorder=2)
        for desc in (False, True):
            q = r[r.descent == desc]
            ax.errorbar(q.P_MPa, q.sigma_Sm, yerr=[q.sigma_Sm - q.sigma_lo, q.sigma_hi - q.sigma_Sm],
                        fmt=mk, color=col, mfc='#fcfcfb' if desc else col,
                        mec=col if desc else '#fcfcfb', ms=6, mew=1.2, elinewidth=1,
                        capsize=0, zorder=3)
        if label in ticks:
            t = ticks[label]
            ax.plot([-6] * len(t), t.sigma_prior_Sm, '>', color='#0b0b0b', ms=4.5, mew=0,
                    zorder=4, clip_on=False)
            ax.plot([-13] * len(t), t.sigma_MC12_Sm, 'D', color='#8a8a86', ms=3.6, mew=0,
                    zorder=4, clip_on=False)
        top = r.loc[r.P_MPa.idxmax()]
        ends.append((top.P_MPa, top.sigma_Sm, f'{label}, {r.T_C.mean():.1f} $^\\circ$C', col))
    return ends


def load_adams_hall():
    """Adams & Hall (1931) 2.865 wt% NaCl, sigma anchored to MC12 at 1 bar."""
    from study_plots import compute_mccleskey_model
    a = pd.read_csv(ADAMS_CSV, comment='#')
    sigma0 = compute_mccleskey_model([ADAMS_MOLAL], [ADAMS_T_C + 273.15], compound='NaCl')[0][0]
    a['P_MPa'] = a.P_bar / 10
    a['sigma_Sm'] = sigma0 / a.R_over_R0
    a['descent'] = a.P_bar.cummax() > a.P_bar
    return a, sigma0


def _plot_adams(ax):
    a, sigma0 = load_adams_hall()
    col = temp_color(ADAMS_T_C)
    up = a[~a.descent].sort_values('P_MPa')
    ax.plot(up.P_MPa, up.sigma_Sm, '-', color=col, lw=1.2, zorder=2)
    for desc in (False, True):
        q = a[a.descent == desc]
        ax.plot(q.P_MPa, q.sigma_Sm, 'X', color=col, ms=7.5,
                mfc='#fcfcfb' if desc else col, mec=col if desc else '#fcfcfb', mew=1.2,
                zorder=3)
    top = up.iloc[-1]
    return (top.P_MPa, top.sigma_Sm,
            fr'NaCl 0.505 m, Adams \& Hall 1931, 25.0 $^\circ$C ($\sigma_0$ {sigma0:.2f} S/m)', col)


def rho_r_water(P_MPa, T_C=25.0):
    """Water density relative to 1 atm at T_C, SeaFreeze 'water1'."""
    from seafreeze.seafreeze import seafreeze
    T = np.array([T_C + 273.15])
    P = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    rho = seafreeze(np.array([P, T], dtype=object), 'water1').rho[:, 0]
    rho0 = seafreeze(np.array([np.array([ATM_MPA]), T], dtype=object), 'water1').rho[0, 0]
    return rho / rho0


def load_fisher_fox_kcl(c0_1e4=200.0):
    """Fisher & Fox (1979) kappa_P/kappa_1 = (Lambda_P/Lambda_1) rho_r(P), 25 C."""
    d = pd.read_csv(FISHER_FOX_CSV, comment='#')
    d = d[np.isclose(d.c0_1e4_molL, c0_1e4)].sort_values('P_atm')
    P = d.P_atm.to_numpy() * ATM_MPA
    ratio = d.Lambda_Scm2mol.to_numpy() / d.Lambda_Scm2mol.iloc[0] * rho_r_water(P)
    return pd.DataFrame({'P_MPa': P, 'ratio': ratio})


def load_fisher_1962_kcl():
    """Fisher (1962): 0.01 M kappa_P/kappa_1 (Table VIII) and 0.02 M Lambda_P/Lambda_1 x Dorsey rho_r."""
    d = pd.read_csv(FISHER_1962_CSV, comment='#')
    one = pd.DataFrame({'P_atm': [1], 'ratio': [1.0]})
    a = d[(d.quantity == 'Lp_rho_over_L1') & np.isclose(d.c0_molL, 0.01, atol=1e-4)]
    a = pd.concat([one, a[['P_atm', 'value']].rename(columns={'value': 'ratio'})])
    lam = d[(d.quantity == 'Lp_over_L1') & np.isclose(d.c0_molL, 0.02)].set_index('P_atm').value
    rho = d[d.quantity == 'rho_r'].set_index('P_atm').value
    b = pd.concat([one, pd.DataFrame({'P_atm': lam.index, 'ratio': (lam * rho.loc[lam.index]).values})])
    out = []
    for df in (a, b):
        df = df.sort_values('P_atm')
        out.append(pd.DataFrame({'P_MPa': df.P_atm.to_numpy() * ATM_MPA, 'ratio': df.ratio.to_numpy()}))
    return out


def load_adams_hall_kcl():
    """Adams & Hall (1931) KCl 0.373 wt%, 25 C: kappa_P/kappa_0 = 1/(R/R0)."""
    a = pd.read_csv(ADAMS_KCL_CSV, comment='#')
    a['P_MPa'] = a.P_bar / 10
    a['ratio'] = 1 / a.R_over_R0
    a['descent'] = (a.index > a.P_bar.idxmax())
    return a


def kcl_literature_table():
    ff = load_fisher_fox_kcl(200.0)
    ff2 = load_fisher_fox_kcl(20.0)
    f62a, f62b = load_fisher_1962_kcl()
    ah = load_adams_hall_kcl()
    parts = [('Fisher and Fox 1979', 0.02, ff), ('Fisher and Fox 1979', 0.002, ff2),
             ('Fisher 1962 (Table VIII)', 0.01, f62a), ('Fisher 1962 (Lambda ratio x Dorsey rho_r)', 0.02, f62b),
             ('Adams and Hall 1931', 0.0502, ah)]
    t = pd.concat([d[['P_MPa', 'ratio']].assign(source=s, c=c, T_C=LIT_T_C) for s, c, d in parts],
                  ignore_index=True)
    return t[['source', 'c', 'T_C', 'P_MPa', 'ratio']]


def _plot_kcl_ratio(ax, rec):
    """Lower panel: kappa(P)/kappa(0.1 MPa) for our KCl standards and literature KCl."""
    for label, _, comp, mk, _ in RUNS_2026:
        if comp != 'KCl':
            continue
        r = rec[rec.run == label].copy()
        col = temp_color(r.T_C.mean())
        r['ratio'] = r.R_amb_first / r.R
        ax.plot(r.P_MPa, r.ratio, '-', color=col, lw=0.8, zorder=2)
        for desc in (False, True):
            q = r[r.descent == desc]
            ax.plot(q.P_MPa, q.ratio, mk, color=col, mfc='#fcfcfb' if desc else col,
                    mec=col if desc else '#fcfcfb', ms=5, mew=1.0, zorder=3)
    dk = '#0b0b0b'
    ff = load_fisher_fox_kcl(200.0)
    ff2 = load_fisher_fox_kcl(20.0)
    f62a, f62b = load_fisher_1962_kcl()
    ah = load_adams_hall_kcl()
    ends = []
    ax.plot(ff.P_MPa, ff.ratio, '-^', color=dk, lw=1.1, ms=5, zorder=4)
    ax.plot(ff2.P_MPa, ff2.ratio, '--', color='#6b6b68', lw=1, zorder=3)
    ax.plot(f62a.P_MPa, f62a.ratio, 'v', color='#6b6b68', ms=6, zorder=4)
    ax.plot(f62b.P_MPa, f62b.ratio, 'v', color='#6b6b68', mfc='#fcfcfb', ms=6, zorder=4)
    up = ah[~ah.descent]
    ax.plot(up.P_MPa, up.ratio, '-', color=dk, lw=0.8, zorder=2)
    for desc in (False, True):
        q = ah[ah.descent == desc]
        ax.plot(q.P_MPa, q.ratio, 'X', color=dk, ms=7, mfc='#fcfcfb' if desc else dk,
                mec=dk if desc else '#fcfcfb', mew=1.0, zorder=4)
    deg = r'$^\circ$C'
    items = [(ff.P_MPa.iloc[-1], ff.ratio.iloc[-1], r'KCl 0.02 M, Fisher \& Fox 1979, 25 ' + deg, dk, '-'),
             (ff2.P_MPa.iloc[-1], ff2.ratio.iloc[-1], r'KCl 0.002 M, Fisher \& Fox 1979, 25 ' + deg, '#6b6b68', '--'),
             (f62a.P_MPa.iloc[-1], f62a.ratio.iloc[-1], r'KCl 0.01 M, Fisher 1962, 25 ' + deg, '#6b6b68', '-'),
             (f62b.P_MPa.iloc[-1], f62b.ratio.iloc[-1],
              r'KCl 0.02 M, Fisher 1962 ($\Lambda_P/\Lambda_1\times\rho_r$), 25 ' + deg, '#6b6b68', '-'),
             (up.P_MPa.iloc[-1], up.ratio.iloc[-1], r'KCl 0.05 m, Adams \& Hall 1931, 25 ' + deg, dk, '-')]
    label_ends(ax, items, 310, gap=0.011)
    ax.set_ylim(0.98, 1.15)
    ax.set_ylabel(r'$\sigma(P)/\sigma(0.1\,\mathrm{MPa})$')
    ax.set_xlabel('Pressure (MPa)')
    ax.grid(True, color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.015, 0.97, 'literature: dilute KCl (0.0005--0.05 mol/kg); ours: 0.70 mol/kg standard\n'
            'filled: compression, hollow: decompression; colour: T of our runs (as above)',
            transform=ax.transAxes, fontsize=7.4, va='top', color='#52514e')


def _plot_pan(ax, st, rec):
    """Pan et al. (2021) on each measured NaCl isotherm."""
    P = np.linspace(0.1, 700, 300)
    iso = [(0.5, T, temp_color(T)) for T in
           st[st.excluded == ''].groupby('run').T_C.mean()
           if not 18 < T < 19.5]  # one curve for the three ~20 C days
    iso += [(0.9, r.T_C.mean(), temp_color(r.T_C.mean()))
            for _, r in rec[rec.comp == 'NaCl'].groupby('run')]
    seen, ends = set(), []
    for m, T, col in iso:
        key = (m, round(T))
        if key in seen:
            continue
        seen.add(key)
        T_K = T + 273.15
        Pm = P if m == 0.5 else P[P <= 215]
        sig = pan.sigma_vs_P(m, T_K, Pm)
        inside = pan.in_envelope(Pm, T_K)
        ax.plot(Pm, sig, '--', color=col, lw=1, alpha=0.75, zorder=1)
        ax.plot(np.where(inside, Pm, np.nan), np.where(inside, sig, np.nan), '-',
                color=col, lw=2.6, alpha=0.35, zorder=1)
        ends.append((Pm[-1], sig[-1], f'Pan et al. 2021, {m} m, {T:.1f} $^\\circ$C', col, '--'))
    return ends


def main():
    os.makedirs(OUT, exist_ok=True)
    _style()
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
    kcl_literature_table().round(5).to_csv(os.path.join(OUT, 'kcl_literature_ratio.csv'),
                                           index=False)
    plot(st, os.path.join(OUT, 'nacl_0p5m_sigma_P.png'), recent=recent)
    plot(st, os.path.join(OUT, 'nacl_0p5m_sigma_P_excluded.png'), show_excluded=True, recent=recent, ex26=ex26)
    print(f'{(st.excluded == "").sum()} steps kept, {(st.excluded != "").sum()} excluded -> {OUT}')


if __name__ == '__main__':
    main()
