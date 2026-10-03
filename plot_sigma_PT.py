#!/usr/bin/env python
"""
Generate 2D sigma(P,T) maps from HiPOZ conductivity data.

Loads curated conductivities from data/<date>/zAnalysis<date>.csv files and
draws the P-T plane map (gamryPlots.plot_conductivity_PT): points colored by
conductivity with interpolated iso-conductivity contours, separating pressure
from temperature dependence.

Dates without a zAnalysis file can be passed via --raw-dates: their Gamry
sweeps are circuit-fitted (CPE, same settings as gamry_HiPOZ.py) and converted
to PROVISIONAL conductivities using the cell constant implied by the curated
dates (K_cell = sigma * R per curated row, averaged), or --k-cell if given.
Provisional points are overlaid as red-edged triangles and written to a CSV;
a companion figure shows their time-ordered sigma to reveal plateau steps
(composition/condition changes within a day).

Examples
--------
KCl standard study (curated Aug-Sep 2026 dates + raw Sep 22-25 uploads):

  python plot_sigma_PT.py \
      --dates 20260818 20260820 20260827 20260828 20260829 20260831 \
              20260901 20260902 20260903 20260904 20260908 20260909 20260910 \
      --raw-dates 20260922 20260923 20260924 20260925 \
      --sigma-ref 8.0

Curated dates only:

  python plot_sigma_PT.py --dates 20260818 20260827 --sigma-ref 8.0

Outputs land in --out-dir (default sigma_PT_plots/): sigma_PT_map.pdf/.png,
and with raw dates also raw_sigma_timeseries.pdf/.png and
provisional_sigma.csv.
"""
import argparse
import logging
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger('HiPOZ')

F_RANGE_HZ = 1e3 * np.array([10, 100])  # same fit band as gamry_HiPOZ.py


def load_curated(dates, data_dir='data'):
    """Load measurement rows from zAnalysis<date>.csv for each date."""
    frames = []
    for d in dates:
        path = Path(data_dir) / d / f'zAnalysis{d}.csv'
        if not path.exists():
            raise FileNotFoundError(
                f'{path} not found - curated dates need a zAnalysis CSV; '
                f'pass uncurated dates via --raw-dates instead')
        df = pd.read_csv(path)
        df['date'] = d
        frames.append(df)
    cur = pd.concat(frames, ignore_index=True)
    cur = cur[(cur['type'] == 'measurement')
              & (cur['exclude'].isna() | (cur['exclude'] == ''))].copy()
    for c in ['P_MPa', 'T_K', 'conductivity_Sm', 'Z_Ohm']:
        cur[c] = pd.to_numeric(cur[c], errors='coerce')
    cur = cur.dropna(subset=['P_MPa', 'T_K', 'conductivity_Sm'])
    return cur


def implied_k_cell(cur):
    """Cell constant implied by curated rows: K = sigma * R, averaged."""
    k = (cur['conductivity_Sm'] * cur['Z_Ohm']).dropna()
    if len(k) == 0:
        raise ValueError('No curated rows with both sigma and Z to infer K_cell; '
                         'pass --k-cell explicitly')
    per_date = cur.assign(K=cur['conductivity_Sm'] * cur['Z_Ohm']) \
                  .groupby('date')['K'].mean()
    log.info(f'Implied K_cell per date (1/m): '
             + ', '.join(f'{d}={v:.2f}' for d, v in per_date.items()))
    return float(k.mean())


def fit_raw_dates(dates, k_cell, data_dir='data', max_files=None):
    """CPE-fit raw Gamry sweeps and convert R -> provisional sigma via k_cell."""
    from gamryTools import Solution  # deferred: heavy import chain
    rows = []
    for d in dates:
        files = sorted(glob(str(Path(data_dir) / d / 'Conductivity*' / '*.txt')))
        if max_files is not None:
            files = files[:max_files]
        if not files:
            log.warning(f'No raw sweep files found for {d}')
        for fpath in files:
            try:
                sol = Solution(cmap_name='viridis')
                sol.load_file(fpath)
                sol.fit_circuit(circ_type='CPE', print_circuit=False,
                                basin_hopping=False, f_range_hz=F_RANGE_HZ)
                rows.append({'date': d, 'file': fpath, 'P_MPa': sol.P_MPa,
                             'T_K': sol.T_K, 'R_ohm': sol.Rcalc_ohm,
                             'R_unc_ohm': sol.Runc_ohm,
                             'sigma_Sm': k_cell / sol.Rcalc_ohm})
            except Exception as e:
                log.error(f'Fit failed for {fpath}: {e}')
    return pd.DataFrame(rows)


def sanity_filter(raw, sigma_min, sigma_max, max_r_unc_frac):
    ok = ((raw['sigma_Sm'] > sigma_min) & (raw['sigma_Sm'] < sigma_max)
          & (raw['R_unc_ohm'] / raw['R_ohm'] < max_r_unc_frac))
    dropped = (~ok).sum()
    if dropped:
        log.warning(f'Sanity cuts dropped {dropped} of {len(raw)} provisional '
                    f'points ({sigma_min}<sigma<{sigma_max} S/m, '
                    f'R unc <{max_r_unc_frac:.0%})')
    return raw[ok].copy()


def make_map(cur, raw, sigma_ref, title, out_dir, xtns):
    import matplotlib.pyplot as plt
    from gamryPlots import plot_conductivity_PT
    fig = plt.figure(figsize=(9.5, 6.5))
    ax, sc = plot_conductivity_PT(fig, cur['P_MPa'], cur['T_K'],
                                  cur['conductivity_Sm'],
                                  sigma_ref_Sm=sigma_ref, title=title)
    if raw is not None and len(raw) > 0 and sc is not None:
        vmin = float(cur['conductivity_Sm'].min())
        vmax = float(cur['conductivity_Sm'].max())
        ax.scatter(raw['P_MPa'], raw['T_K'] - 273.15,
                   c=raw['sigma_Sm'].clip(vmin, vmax), cmap='viridis',
                   vmin=vmin, vmax=vmax, marker='^', s=55,
                   edgecolors='r', linewidths=0.8, zorder=4,
                   label='provisional (raw fit)')
        ax.legend(loc='best', fontsize=8)
    paths = []
    for xtn in xtns:
        p = out_dir / f'sigma_PT_map.{xtn}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        paths.append(p)
    plt.close(fig)
    return paths


def make_raw_timeseries(raw, sigma_ref, out_dir, xtns):
    import matplotlib.pyplot as plt
    raw = raw.copy()
    raw['order'] = raw.groupby('date').cumcount()
    dates = sorted(raw['date'].unique())
    fig, axes = plt.subplots(1, len(dates), figsize=(3.3 * len(dates), 4),
                             sharey=True, squeeze=False)
    for ax, d in zip(axes[0], dates):
        g = raw[raw['date'] == d].sort_values('file')
        ax.plot(range(len(g)), g['sigma_Sm'], 'o-', ms=4, lw=0.8)
        if sigma_ref is not None:
            ax.axhline(sigma_ref, color='r', lw=1, ls='--')
        ax.set_title(d)
        ax.set_xlabel('sweep (time order)')
        ax.grid(True, ls=':', alpha=0.6)
    axes[0][0].set_ylabel(r'provisional $\sigma$ (S/m)')
    fig.suptitle('Raw sweeps: time-ordered provisional conductivity')
    fig.tight_layout()
    paths = []
    for xtn in xtns:
        p = out_dir / f'raw_sigma_timeseries.{xtn}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dates', nargs='+', required=True,
                        help='Curated dates (must have data/<date>/zAnalysis<date>.csv)')
    parser.add_argument('--raw-dates', nargs='+', default=[],
                        help='Dates to circuit-fit from raw sweeps (no zAnalysis needed)')
    parser.add_argument('--k-cell', type=float, default=None,
                        help='Cell constant (1/m) for raw dates; default: implied by curated data')
    parser.add_argument('--sigma-ref', type=float, default=None,
                        help='Reference conductivity (S/m) to highlight, e.g. 8.0')
    parser.add_argument('--title', default=r'Conductivity in the P--T plane')
    parser.add_argument('--out-dir', default='sigma_PT_plots')
    parser.add_argument('--xtn', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png'], help='Output formats')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--sigma-min', type=float, default=1.0,
                        help='Sanity cut: discard provisional sigma below this')
    parser.add_argument('--sigma-max', type=float, default=15.0,
                        help='Sanity cut: discard provisional sigma above this')
    parser.add_argument('--max-r-unc', type=float, default=0.2,
                        help='Sanity cut: discard fits with fractional R uncertainty above this')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Fit at most this many files per raw date (for quick tests)')
    parser.add_argument('--no-tex', action='store_true',
                        help='Disable LaTeX text rendering (for machines without TeX)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    import matplotlib
    if matplotlib.get_backend().lower() != 'agg' and not sys.stdout.isatty():
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F401  (backend must be set first)
    import gamryPlots  # noqa: F401  (sets rcParams, incl. usetex)
    if args.no_tex:
        plt.rcParams['text.usetex'] = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cur = load_curated(args.dates, args.data_dir)
    log.info(f'{len(cur)} curated measurements from {len(args.dates)} date(s); '
             f'P {cur.P_MPa.min():.0f}-{cur.P_MPa.max():.0f} MPa, '
             f'T {cur.T_K.min():.1f}-{cur.T_K.max():.1f} K, '
             f'sigma {cur.conductivity_Sm.min():.2f}-{cur.conductivity_Sm.max():.2f} S/m')

    raw = None
    if args.raw_dates:
        k_cell = args.k_cell if args.k_cell is not None else implied_k_cell(cur)
        log.info(f'Using K_cell = {k_cell:.2f} 1/m for provisional conductivities '
                 f'- verify this applies to the raw dates (cell unchanged?)')
        raw = fit_raw_dates(args.raw_dates, k_cell, args.data_dir, args.max_files)
        log.info(f'Fitted {len(raw)} raw sweeps from {len(args.raw_dates)} date(s)')
        raw = sanity_filter(raw, args.sigma_min, args.sigma_max, args.max_r_unc)
        csv_path = out_dir / 'provisional_sigma.csv'
        raw.to_csv(csv_path, index=False)
        log.info(f'Wrote {csv_path}')

    paths = make_map(cur, raw, args.sigma_ref, args.title, out_dir, args.xtn)
    if raw is not None and len(raw) > 0:
        paths += make_raw_timeseries(raw, args.sigma_ref, out_dir, args.xtn)
    for p in paths:
        log.info(f'Wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
