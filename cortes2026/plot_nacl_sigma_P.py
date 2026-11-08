#!/usr/bin/env python3
"""
Clean sigma(P) for the Gamry high-pressure cell.

Left: isotherms of the Aug 2025 Cortes 0.5 mol/kg NaCl runs.
Right: the 2026 runs with the correct trend, each as sigma(P) / sigma at
ambient pressure of the same run: 0.9 mol/kg (5 wt%) NaCl on Oct 7-8
(30-min steps) and Oct 9 (10-min steps), and the 8 S/m KCl standard on
Oct 14-15, Nov 5 and Nov 6. Normalizing within a run cancels the cell
constant, which is unresolved for Oct 7-9 (open question 2026-10-19).
Left out (Steve, 2026-11-08): the Aug 18 - Sep 10 2026 KCl runs, stepped too
fast for the trend with pressure; also Oct 6 NaCl (5-10 min steps, up/down
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
os.chdir(REPO)
import cortes_data_processing as cdp  # noqa: E402
from gamryTools import Solution  # noqa: E402

OUT = os.path.join(HERE, 'nacl_sigma_P')
MAX_GAP_MPA = 150
# Isotherms joined across a wider gap (Steve, 2026-11-08)
ALWAYS_CONNECT = {('20250820', 20)}
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
    ('20250818', 20): ('#1baf7a', 's'),
    ('20250820', 20): ('#1baf7a', '^'),
    ('20250813', 20): ('#1baf7a', 'o'),
    ('20250820', 35): ('#a8322f', 'o'),
}

# 2026 runs: label, folders (fitted from the raw sweeps), colour. Colours
# validated as their own palette (dataviz validate_palette, light surface).
RUNS_2026 = [
    ('NaCl 0.9 m, Oct 7-8, 23 C, 30-min steps', ['20261007', '20261008'], '#2a78d6'),
    ('NaCl 0.9 m, Oct 9, 23 C, 10-min steps', ['20261009'], '#eb6834'),
    ('KCl 8 S/m std, Oct 14-15, 23 C', ['20261014', '20261015'], '#1baf7a'),
    ('KCl 8 S/m std, Nov 5, 25 C', ['20261105'], '#eda100'),
    ('KCl 8 S/m std, Nov 6, 20 C', ['20261106'], '#e87ba4'),
]
F_RANGE_HZ = 1e3 * np.array([10, 100])


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


def load_2026():
    """Step means of R for each 2026 run, as sigma/sigma_ambient = R_amb/R."""
    rows = []
    for label, folders, _ in RUNS_2026:
        sw = []
        for folder in folders:
            skip = _excluded_files(folder)
            for f in sorted(glob(os.path.join(REPO, 'data', folder, 'Conductivity*', '*.txt'))):
                if os.path.basename(f) in skip:
                    continue
                s = Solution(cmap_name='viridis')
                s.load_file(f)
                s.fit_circuit(circ_type='CPE', initial_guess=None, print_circuit=False,
                              basin_hopping=False, multiproc=False, f_range_hz=F_RANGE_HZ)
                sw.append(dict(time=s.time, P_MPa=s.P_MPa, T_K=s.T_K, R=s.Rcalc_ohm))
        sw = pd.DataFrame(sw).sort_values('time').reset_index(drop=True)
        new = (sw.time.diff().dt.total_seconds() > 240) | (sw.P_MPa.diff().abs() > 10)
        sw['step'] = new.cumsum()
        st = sw.groupby('step').agg(time=('time', 'first'), P_MPa=('P_MPa', 'mean'),
                                    T_K=('T_K', 'mean'), R=('R', 'mean'),
                                    R_min=('R', 'min'), R_max=('R', 'max'), n=('R', 'size'))
        R_amb = st.R.iloc[0]  # every run starts at ambient pressure
        st['ratio'] = R_amb / st.R
        st['ratio_lo'] = R_amb / st.R_max
        st['ratio_hi'] = R_amb / st.R_min
        st['descent'] = st.P_MPa.cummax() > st.P_MPa + 5
        st['run'] = label
        rows.append(st)
    return pd.concat(rows, ignore_index=True)


def _style():
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
        'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
        'xtick.color': '#52514e', 'ytick.color': '#52514e',
    })


def plot(st, path, show_excluded=False, recent=None):
    if recent is None:
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
    else:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                      gridspec_kw={'width_ratios': [1.25, 1]})
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
            if P[i + 1] - P[i] <= MAX_GAP_MPA or run in ALWAYS_CONNECT:
                ax.plot(P[i:i + 2], S[i:i + 2], '-', color=col, lw=1.5, zorder=2)
        ax.errorbar(r.P_MPa, r.sigma_Sm, yerr=yerr, fmt=mk, color=col, ms=6.5,
                    mec='#fcfcfb', mew=1.2, elinewidth=1, capsize=0, zorder=3,
                    label=label + r' $^\circ$C')
    if show_excluded:
        ex = st[st.excluded != '']
        ax.plot(ex.P_MPa, ex.sigma_Sm, 'x', color='#0b0b0b', ms=7, mew=1.5,
                label='excluded', zorder=4)
        for _, r in ex.iterrows():
            ax.annotate(f"{r.start:%m/%d %H:%M}", (r.P_MPa, r.sigma_Sm),
                        textcoords='offset points', xytext=(6, -2),
                        fontsize=6.5, color='#52514e')
    ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel(r'$\sigma$ (S/m)')
    ax.set_xlim(-15, 720)
    ax.grid(True, color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(r'0.5 mol/kg NaCl, Gamry high-pressure cell (Cortes, Aug 2025)',
                 fontsize=11)
    if recent is None:
        ax.legend(title=r'run, $T$', fontsize=8.5, title_fontsize=9, frameon=False,
                  loc='upper left', bbox_to_anchor=(1.01, 1.0))
    else:
        ax.legend(title=r'run, $T$', fontsize=8, title_fontsize=8.5, frameon=False,
                  loc='lower center', ncol=3)
        _plot_recent(ax2, recent)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if path.endswith('.png') and not show_excluded:
        fig.savefig(path[:-4] + '.pdf')
    plt.close(fig)


def _plot_recent(ax, rec):
    for label, _, col in RUNS_2026:
        r = rec[rec.run == label]
        ax.plot(r.P_MPa, r.ratio, '-', color=col, lw=1.5, zorder=2)
        for desc, mfc in ((False, col), (True, '#fcfcfb')):
            q = r[r.descent == desc]
            ax.errorbar(q.P_MPa, q.ratio, yerr=[q.ratio - q.ratio_lo, q.ratio_hi - q.ratio],
                        fmt='o', color=col, mfc=mfc, mec=col if desc else '#fcfcfb',
                        ms=6, mew=1.2, elinewidth=1, capsize=0, zorder=3,
                        label=None if desc else label)
    ax.plot([], [], 'o', color='#52514e', mfc='#fcfcfb', ms=6, label='hollow: decompression')
    ax.axhline(1, color='#9a9a9a', lw=0.8, zorder=1)
    ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel(r'$\sigma(P)\,/\,\sigma(P_\mathrm{amb})$, same run')
    ax.set_xlim(-8, 215)
    ax.grid(True, color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title('2026 runs (Diaz), normalized to ambient', fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc='upper left')


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
    recent = load_2026()
    recent.round(5).to_csv(os.path.join(OUT, 'sigma_P_2026_steps.csv'), index=False)
    plot(st, os.path.join(OUT, 'nacl_0p5m_sigma_P.png'), recent=recent)
    plot(st, os.path.join(OUT, 'nacl_0p5m_sigma_P_excluded.png'), show_excluded=True)
    print(f'{(st.excluded == "").sum()} steps kept, {(st.excluded != "").sum()} excluded -> {OUT}')


if __name__ == '__main__':
    main()
