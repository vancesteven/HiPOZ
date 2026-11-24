#!/usr/bin/env python3
"""
Data sources for the conductivity-pressure figures, in one tidy format.

Purpose
-------
Every curve on the sigma(P) figures comes from one function here, returning a
pandas DataFrame with the SAME columns, in ABSOLUTE units (S/m). No plotting
and no slow sweep fitting happens in this module; it only reads CSV files that
plot_nacl_sigma_P.py has already written, plus the small literature tables.

Pipeline (see nacl_sigma_P/README.md for the order of commands)
    raw Gamry sweeps --(plot_nacl_sigma_P.py, ~30 s)--> step CSVs
    step CSVs + literature --(this module)--> tidy DataFrames
    tidy DataFrames --> plot_nacl_sigma_P.py   (supplement, everything)
                    --> plot_sigma_P_main.py   (main text, best assessment)

Data model (columns of every DataFrame returned by a source function)
    source     who produced the number ('Cortes 2025', 'Diaz 2026', 'Fisher and
               Fox 1979', ...)
    run        unique key of one isotherm / one literature series
    run_short  short human label ('Aug 13', 'Nov 5', '0.02 M')
    solute     'NaCl' or 'KCl'
    molality   mol/kg water (literature given in mol/L: approx. equal, see below)
    T_C        temperature, deg C (mean over the step)
    P_MPa      pressure, MPa (mean over the step)
    sigma_Sm   conductivity, S/m
    sigma_lo, sigma_hi
               lowest / highest sweep inside the step (ours); equal to sigma_Sm
               for literature
    descent    True on the decompression leg
    excluded   '' if the step is kept, otherwise the reason it is excluded
    K_source   where the cell constant came from (ours); the anchor of the
               absolute scale (literature)
    time       start of the step (ours), NaT for literature
    ratio      sigma / sigma at the run's first 1-bar step (added by with_ratio)
Rows are in measurement order within a run.

Our own steps (Nov 17-19 2026: six 0.9 mol/kg NaCl isotherms, 24.8 to 2 C,
    all with ONE K from the first 1-bar step of the 24.8 C isotherm against
    McCleskey 2011, so the 1-bar offsets of the colder isotherms are visible in
    the supplement; the main figure anchors each isotherm separately. Nov 19
    from 16:44 is a KCl calibration outside the vessel: recorded in
    diaz_hp_sweep_fits.csv with an exclusion reason, not a run here.)
    Each step is a group of consecutive sweeps (< 4 min apart at the same
    nominal pressure): sigma is the mean, sigma_lo/sigma_hi the extremes. Cell
    constants K (sigma = K / R) come from different places in different runs;
    K_source says which (details in plot_nacl_sigma_P.py's docstring).

Literature
    Fisher and Fox (1979): absolute. Lambda(P) c0 rho_r(P) / 1000, see
        fisher_fox_kcl().
    Fisher (1962): ratios only; made absolute with Fisher and Fox's 1-atm
        values, see fisher_1962_kcl().
    Adams and Hall (1931): ratios only (R/R0); made absolute with a 1-bar sigma
        from McCleskey et al. (2012), see adams_hall_nacl()/adams_hall_kcl().
    Pan et al. (2021): a model, pan_nacl().

Reference 1-bar conductivity (sigma_ref_1bar) and the anchoring of a run to it
(anchor_to_1bar) are documented at those functions.

Run from the repository root (importing gamryTools pulls in PlanetProfile,
which asks for UserConfigs in the working directory; this module therefore
changes into the repository root on import, like the plotting scripts).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)
import vessel_1bar_check as vbc  # noqa: E402
import pan2021_nacl as pan  # noqa: E402

OUT = os.path.join(HERE, 'nacl_sigma_P')
STEPS_2025_CSV = os.path.join(OUT, 'nacl_0p5m_sigma_P_steps.csv')
STEPS_2026_CSV = os.path.join(OUT, 'sigma_P_2026_steps.csv')
SWEEPS_2026_CSV = os.path.join(HERE, 'diaz', 'diaz_hp_sweep_fits.csv')
ADAMS_NACL_CSV = os.path.join(HERE, 'literature', 'adams_hall_1931_nacl_2p865.csv')
FISHER_FOX_CSV = os.path.join(HERE, 'literature', 'fisher_fox_1979_kcl_25C.csv')
FISHER_1962_CSV = os.path.join(HERE, 'literature', 'fisher_1962_kcl_25C.csv')
# ThermoLit is a sibling checkout (../ThermoLit); override with THERMOLIT_DIR
THERMOLIT = os.environ.get('THERMOLIT_DIR', os.path.join(os.path.dirname(REPO), 'ThermoLit'))
ADAMS_KCL_CSV = os.path.join(THERMOLIT, 'data', 'adams_hall_1931',
                             'table1_kcl_0.3730wt_25.00C.csv')

ATM_MPA = 0.101325
AMBIENT_MAX_MPA = 3.0          # steps below this are '1 bar' (as vessel_1bar_check)
LIT_T_C = 25.0                 # all literature below is at 25.00 C
M_NACL, M_KCL = 0.05844, 0.074551   # kg/mol
ADAMS_NACL_MOLAL = 0.02865 / (1 - 0.02865) / M_NACL        # 2.8650 wt% -> 0.5047 mol/kg
ADAMS_KCL_MOLAL = 0.00373 / (1 - 0.00373) / M_KCL          # 0.3730 wt% -> 0.0502 mol/kg

COLUMNS = ['source', 'run', 'run_short', 'solute', 'molality', 'T_C', 'P_MPa', 'sigma_Sm',
           'sigma_lo', 'sigma_hi', 'descent', 'excluded', 'K_source', 'time']

# Our 2026 runs, keyed by the run label used in plot_nacl_sigma_P.RUNS_2026
# (the step CSV's 'run' column): short label, solute, molality (mol/kg),
# where the cell constant came from.
_K_OCT14 = 'Oct 14 8 S/m KCl standard (estimate, measured 5 days later)'
_K_OWN = "own first 1-bar triplet = 8 S/m bottle value"
_K_NOV = 'common Nov K 120.617 (median of settled 1-bar steps vs bottle table)'
_K_PRIOR = 'first 1-bar step vs McCleskey (2011), soldered connection broke (PI 2026-11-18)'
_K_NOV17 = ('ONE K for Nov 17-19: first 1-bar step (Nov 17 09:46, 24.8 C) vs McCleskey (2011); '
            'connection repaired, PI 2026-11-20')
RUN_META_2026 = {
    'NaCl 0.9 m, Oct 7-8, 30-min steps': ('Oct 7-8', 'NaCl', 0.9006, _K_OCT14),
    'NaCl 0.9 m, Oct 9, 10-min steps': ('Oct 9', 'NaCl', 0.9006, _K_OCT14),
    'KCl 8 S/m std, Oct 14-15': ('Oct 14-15', 'KCl', vbc.KCL_STD_MOLAL, _K_OWN),
    'KCl 8 S/m std, Nov 5': ('Nov 5', 'KCl', vbc.KCL_STD_MOLAL, _K_NOV),
    'KCl 8 S/m std, Nov 6': ('Nov 6', 'KCl', vbc.KCL_STD_MOLAL, _K_NOV),
    'KCl 8 S/m std, Nov 9': ('Nov 9', 'KCl', vbc.KCL_STD_MOLAL, _K_NOV),
    'KCl 8 S/m std, Nov 10': ('Nov 10', 'KCl', vbc.KCL_STD_MOLAL, _K_NOV),
    'KCl 8 S/m std, Nov 11': ('Nov 11', 'KCl', vbc.KCL_STD_MOLAL, _K_NOV),
    'KCl 8 S/m std, Nov 11-12': ('Nov 11-12', 'KCl', vbc.KCL_STD_MOLAL, _K_NOV),
    'NaCl 0.9 m, Nov 16': ('Nov 16', 'NaCl', 0.902, _K_PRIOR),
    'NaCl 0.9 m, Nov 17, 24.8 C': ('Nov 17', 'NaCl', 0.902, _K_NOV17),
    'NaCl 0.9 m, Nov 17, 20.5 C': ('Nov 17', 'NaCl', 0.902, _K_NOV17),
    'NaCl 0.9 m, Nov 17-18, 16.2 C': ('Nov 17-18', 'NaCl', 0.902, _K_NOV17),
    'NaCl 0.9 m, Nov 18, 11.5 C': ('Nov 18', 'NaCl', 0.902, _K_NOV17),
    'NaCl 0.9 m, Nov 18-19, 7.0 C': ('Nov 18-19', 'NaCl', 0.902, _K_NOV17),
    'NaCl 0.9 m, Nov 19, 2 C': ('Nov 19', 'NaCl', 0.902, _K_NOV17),
}


# ---------------------------------------------------------------- our steps
def _descent(P):
    """True once the pressure has dropped > 5 MPa below its running maximum."""
    P = np.asarray(P, dtype=float)
    return np.maximum.accumulate(P) > P + 5


def our_steps_2025(include_excluded=True):
    """Cortes, Aug 2025: 0.5 mol/kg NaCl isotherms in the small Bollengier vessel.

    Read from nacl_sigma_P/nacl_0p5m_sigma_P_steps.csv (written by
    plot_nacl_sigma_P.py). sigma uses the cell constants in the 2025 zAnalysis
    configs. Isotherms are keyed by (date, bath setpoint): the three ~20 C days
    differ by date. The Aug 19 16:35 600 MPa step belongs to the Aug 20 20 C
    isotherm (same loading, same setpoint). 'descent' is True only for the
    hasty Aug 19 decompression steps, which are all excluded; no kept 2025 step
    is a decompression step.
    """
    d = pd.read_csv(STEPS_2025_CSV, keep_default_na=False)
    runs = d.run.map(lambda s: eval(s, {'__builtins__': {}}))  # "('20250813', 20)"
    out = pd.DataFrame({
        'source': 'Cortes 2025',
        'run': ['NaCl 0.5 m, Aug %d, %d C' % (int(r[0][6:]), r[1]) for r in runs],
        'run_short': ['Aug %d' % int(r[0][6:]) for r in runs],
        'solute': 'NaCl', 'molality': 0.5, 'T_C': d.T_C, 'P_MPa': d.P_MPa,
        'sigma_Sm': d.sigma_Sm, 'sigma_lo': d.sigma_min, 'sigma_hi': d.sigma_max,
        'descent': d.excluded.str.contains('decompression'),
        'excluded': d.excluded,
        'K_source': 'cell constants of the 2025 zAnalysis configs',
        'time': pd.to_datetime(d.start)})
    out = out.sort_values(['run', 'time']).reset_index(drop=True)
    return out if include_excluded else out[out.excluded == ''].reset_index(drop=True)


def our_steps_2026(include_excluded=True):
    """Diaz, Oct-Nov 2026: NaCl 0.9 mol/kg and the 0.70 mol/kg KCl standard,
    in the large vessel.

    Kept steps come from sigma_P_2026_steps.csv (step means of sigma = K/R with
    min-max of the sweeps). Excluded sweeps (zAnalysis config exclusions and
    the provisional +/-12 ohm CPE fits) come from diaz_hp_sweep_fits.csv and are
    grouped into steps the same way plot_nacl_sigma_P.load_2026 does: sweeps
    < 4 min apart and < 10 MPa apart. The Oct 6 run (all sweeps 'left out') is
    not part of the figures and is skipped.
    """
    k = pd.read_csv(STEPS_2026_CSV, parse_dates=['time'])
    rows = []
    for run, s in k.groupby('run', sort=False):
        short, solute, m, ksrc = RUN_META_2026[run]
        rows.append(pd.DataFrame({
            'source': 'Diaz 2026', 'run': run, 'run_short': short, 'solute': solute,
            'molality': m, 'T_C': s.T_C, 'P_MPa': s.P_MPa, 'sigma_Sm': s.sigma_Sm,
            'sigma_lo': s.sigma_lo, 'sigma_hi': s.sigma_hi, 'descent': s.descent.astype(bool),
            'excluded': '', 'K_source': ksrc, 'time': s.time}))
    if include_excluded:
        sw = pd.read_csv(SWEEPS_2026_CSV, parse_dates=['time'], keep_default_na=False)
        sw['excluded'] = sw.excluded.astype(str)
        for c in ('sigma_Sm', 'K_cell'):  # blank for the out-of-vessel sweeps
            sw[c] = pd.to_numeric(sw[c], errors='coerce')
        for run, s in sw[sw.run.isin(RUN_META_2026)].groupby('run', sort=False):
            s = s.sort_values('time')
            desc = pd.Series(_descent(s.P_MPa), index=s.index)
            ex = s[s.excluded != '']
            if ex.empty:
                continue
            new = (ex.time.diff().dt.total_seconds() > 240) | (ex.P_MPa.diff().abs() > 10)
            ex = ex.assign(step=new.cumsum(), descent=desc[ex.index])
            g = ex.groupby('step')
            short, solute, m, ksrc = RUN_META_2026[run]
            rows.append(pd.DataFrame({
                'source': 'Diaz 2026', 'run': run, 'run_short': short, 'solute': solute,
                'molality': m, 'T_C': g.T_K.mean() - 273.15, 'P_MPa': g.P_MPa.mean(),
                'sigma_Sm': g.sigma_Sm.mean(), 'sigma_lo': g.sigma_Sm.min(),
                'sigma_hi': g.sigma_Sm.max(), 'descent': g.descent.first(),
                'excluded': g.excluded.first(), 'K_source': ksrc, 'time': g.time.first()}))
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(['run', 'time']).reset_index(drop=True)[COLUMNS]


def first_ambient(run_df):
    """The run's first kept, ascending, 1-bar step (P < 3 MPa), in time order
    (file order for literature). This is the step every ratio and every
    anchor refers to. Returns one row, or raises if the run has none."""
    ok = run_df[(run_df.excluded == '') & (~run_df.descent) & (run_df.P_MPa < AMBIENT_MAX_MPA)]
    if ok.empty:
        raise ValueError(f'no kept ascending 1-bar step in run {run_df.run.iloc[0]!r}')
    if ok.time.notna().all():
        ok = ok.sort_values('time')
    return ok.iloc[0]


def with_ratio(df):
    """Add column 'ratio' = sigma / sigma(first 1-bar step of the same run).
    Runs with no kept ascending 1-bar step (e.g. Aug 14, all excluded) get NaN."""
    df = df.copy()
    df['ratio'] = np.nan
    for run, idx in df.groupby('run', sort=False).groups.items():
        try:
            s0 = first_ambient(df.loc[idx]).sigma_Sm
        except ValueError:
            continue
        df.loc[idx, 'ratio'] = df.loc[idx, 'sigma_Sm'] / s0
    return df


# --------------------------------------------------------- literature (KCl)
def rho_r_water(P_MPa, T_C=LIT_T_C):
    """Density of water relative to 1 atm at T_C (SeaFreeze 'water1')."""
    from seafreeze.seafreeze import seafreeze
    T = np.array([T_C + 273.15])
    P = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    rho = seafreeze(np.array([P, T], dtype=object), 'water1').rho[:, 0]
    rho0 = seafreeze(np.array([np.array([ATM_MPA]), T], dtype=object), 'water1').rho[0, 0]
    return rho / rho0


def _lit_frame(source, run, short, solute, molality, P_MPa, sigma_Sm, descent=None, ksrc=''):
    n = len(P_MPa)
    return pd.DataFrame({
        'source': source, 'run': run, 'run_short': short, 'solute': solute,
        'molality': molality, 'T_C': LIT_T_C, 'P_MPa': np.asarray(P_MPa, dtype=float),
        'sigma_Sm': np.asarray(sigma_Sm, dtype=float), 'sigma_lo': np.asarray(sigma_Sm, dtype=float),
        'sigma_hi': np.asarray(sigma_Sm, dtype=float),
        'descent': np.zeros(n, bool) if descent is None else np.asarray(descent, bool),
        'excluded': '', 'K_source': ksrc, 'time': pd.NaT})[COLUMNS]


def fisher_fox_kcl(c0_molL):
    """Fisher and Fox (1979), J. Solution Chem. 8, 627-634, Table I (p. 629):
    KCl in water at 25 C up to 2000 atm, ABSOLUTE conductivity.

    The table lists molar conductance Lambda (S cm^2 mol^-1) at pressure P for
    solutions of concentration c0 at 1 atm (c0 in mol/L x 1e4 in the file). The
    concentration rises with the water density, c(P) = c0 rho_r(P), so

        kappa(P) = Lambda(P) c0 rho_r(P) / 1000   [S/cm]
        sigma(P) = 100 kappa(P)                    [S/m]

    with rho_r(P) = rho_water(P, 25 C) / rho_water(1 atm, 25 C) from SeaFreeze
    'water1'. Their 1-atm Lambda are absolute: they were normalised to other
    workers' atmospheric-pressure data (refs. 7-8, p. 628), so kappa(1 atm) is
    an absolute conductivity, not just a ratio. Example: Lambda = 138.34 at
    0.02 M gives 0.2767 S/m.

    Molality ~ molarity for these dilute solutions (0.02 M KCl is 0.02005
    mol/kg; the small difference is ignored, it only labels the curve).
    """
    d = pd.read_csv(FISHER_FOX_CSV, comment='#')
    d = d[np.isclose(d.c0_1e4_molL, c0_molL * 1e4)].sort_values('P_atm')
    if d.empty:
        raise ValueError(f'Fisher and Fox have no c0 = {c0_molL} mol/L')
    P = d.P_atm.to_numpy() * ATM_MPA
    sigma = d.Lambda_Scm2mol.to_numpy() * c0_molL * rho_r_water(P) / 1000 * 100
    return _lit_frame('Fisher and Fox 1979', f'KCl {c0_molL:g} M, Fisher and Fox 1979',
                      f'{c0_molL:g} M', 'KCl', c0_molL, P, sigma,
                      ksrc='absolute (Fisher and Fox 1-atm Lambda)')


def fisher_1962_kcl(c0_molL):
    """Fisher (1962), J. Phys. Chem. 66, 1607-1611: KCl at 25 C, 500-2000 atm.

    Fisher gives RATIOS. kappa_P/kappa_1 is
      * c0 = 0.01 M: Table VIII (p. 1609), 'Fisher' column, Lambda_p rho_r /
        Lambda_1;
      * c0 = 0.0005, 0.002, 0.02 M: Table III (p. 1608) Lambda_p/Lambda_1
        times his Dorsey rho_r (Table I).
    (Table III's own 0.01 M row is not used; Table VIII supersedes it.)

    Made absolute with Fisher and Fox's 1-atm conductivity at the same c0:
    sigma(P) = (kappa_P/kappa_1) * sigma_FF(1 atm, c0). Justification: from
    Fisher and Fox's 1-atm kappa, Fisher's implied KCl cell constant is 0.733,
    matching the 0.732 he lists for his other salts at 0.02 M. So the two
    studies agree on the absolute scale and only the 1-atm anchor needs
    borrowing. Molality ~ molarity as for fisher_fox_kcl().
    """
    d = pd.read_csv(FISHER_1962_CSV, comment='#')
    near = lambda c: np.isclose(d.c0_molL, c0_molL, atol=1e-4)
    if np.isclose(c0_molL, 0.01):
        r = d[(d.quantity == 'Lp_rho_over_L1') & near(c0_molL)].set_index('P_atm').value
    else:
        lam = d[(d.quantity == 'Lp_over_L1') & near(c0_molL)].set_index('P_atm').value
        rho = d[d.quantity == 'rho_r'].set_index('P_atm').value
        r = lam * rho.loc[lam.index]
    if r.empty:
        raise ValueError(f'Fisher (1962) has no c0 = {c0_molL} mol/L')
    r = pd.concat([pd.Series({1: 1.0}), r.sort_index()])
    sigma1 = fisher_fox_kcl(c0_molL).sigma_Sm.iloc[0]
    return _lit_frame('Fisher 1962', f'KCl {c0_molL:g} M, Fisher 1962', f'{c0_molL:g} M', 'KCl',
                      c0_molL, r.index.to_numpy() * ATM_MPA, r.to_numpy() * sigma1,
                      ksrc='kappa_P/kappa_1 (Fisher) x Fisher and Fox 1-atm kappa')


def _adams_hall(csv, solute, molality, run, ratio_col='R_over_R0'):
    """Adams and Hall (1931) rows in the order measured (pressure up, then
    down); sigma(P) = sigma0 / (R/R0) with sigma0 from McCleskey et al. (2012)."""
    a = pd.read_csv(csv, comment='#')
    sigma0 = vbc._mc12(LIT_T_C, _ions(solute, molality))
    desc = np.asarray(a.index > a.P_bar.idxmax())
    return _lit_frame('Adams and Hall 1931', run, 'Adams and Hall', solute, molality,
                      a.P_bar.to_numpy() / 10, sigma0 / a[ratio_col].to_numpy(), desc,
                      ksrc='sigma0 = McCleskey et al. (2012) at 25 C; Adams and Hall give R/R0')


def adams_hall_nacl():
    """Adams and Hall (1931), J. Phys. Chem. 35, Table I (cont.), p. 2152:
    NaCl 2.8650 wt% (0.5047 mol/kg), 25.00 C, 1-4189 bar, R/R0 only.

    sigma0 (1 bar) is the McCleskey et al. (2012) model: the PI's 'prudent,
    worst-case' reference. Pressure dependence is Adams and Hall's own.
    """
    return _adams_hall(ADAMS_NACL_CSV, 'NaCl', ADAMS_NACL_MOLAL,
                       'NaCl 0.505 m, Adams and Hall 1931')


def adams_hall_kcl():
    """Adams and Hall (1931), p. 2153: KCl 0.3730 wt% (0.0502 mol/kg), 25.00 C,
    1-2705 bar, R/R0 only. Transcription lives in ../ThermoLit.

    sigma0 from McCleskey et al. (2012). Scale of that anchor's error: MC12 at
    0.0200 mol/kg KCl, 25 C is 0.2788 S/m, +0.8 % above Fisher and Fox's
    0.2767 S/m at 0.02 M (the molality/molarity difference, 0.25 %, is not
    removed).
    """
    return _adams_hall(ADAMS_KCL_CSV, 'KCl', ADAMS_KCL_MOLAL,
                       'KCl 0.050 m, Adams and Hall 1931')


# ------------------------------------------------------------- Pan et al.
def pan_nacl(molality, T_C, P_MPa):
    """Pan et al. (2021) NaCl model (pan2021_nacl.py) along an isotherm.

    Returns the tidy columns plus 'in_envelope': True inside the model's fitted
    range (212-1713 MPa, 233-295 K); outside it the curve is an extrapolation.
    """
    P = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    T_K = T_C + 273.15
    sigma = pan.sigma_vs_P(molality, T_K, P)
    df = _lit_frame('Pan 2021 model', f'NaCl {molality:g} m, Pan 2021, {T_C:.1f} C',
                    'Pan 2021', 'NaCl', molality, P, sigma, ksrc='model')
    df['T_C'] = T_C
    df['in_envelope'] = pan.in_envelope(P, T_K)
    return df


# ------------------------------------------- 1-bar reference and anchoring
# McCleskey (2011), J. Chem. Eng. Data 56, 317-327, Table 3 (mS/cm, 0.1 MPa),
# KCl at 0.5 and 1.0 mol/kg, T = 5, 10, 25, 35, 45, 70, 90 C. Transcribed from
# conductivity-jpl's empirical_conductivity_library.csv; NOT yet checked
# against the paper. (NaCl's table is vessel_1bar_check.MCCLESKEY2011_NACL.)
MCCLESKEY2011_KCL = {  # mol/kg -> kappa (mS/cm)
    0.5: np.array([38.75, 43.75, 59.20, 69.81, 81.07, 112.3, 134.8]),
    1.0: np.array([72.03, 80.84, 108.6, 128.0, 147.7, 202.5, 245.8]),
}
M2011_RANGE_M = (0.5, 1.0)           # mol/kg tabulated
M2011_RANGE_T_C = (5.0, 90.0)


def _ions(solute, m):
    cation = {'NaCl': 'Na_p1', 'KCl': 'K_p1'}[solute]
    return {cation: m, 'Cl_m1': m}


def _sigma_2011_kcl(m, T_C):
    """Same interpolation as vessel_1bar_check.sigma_prior_nacl, for KCl:
    molar conductivity linear in sqrt(m), then a cubic spline in T."""
    (m1, k1), (m2, k2) = sorted(MCCLESKEY2011_KCL.items())
    f = (np.sqrt(m) - np.sqrt(m1)) / (np.sqrt(m2) - np.sqrt(m1))
    lam = (k1 / m1) + f * (k2 / m2 - k1 / m1)
    return float(CubicSpline(vbc.MCCLESKEY2011_T_C, m * lam)(T_C)) / 10.0


def sigma_ref_1bar(solute, molality, T_C):
    """Reference 1-bar conductivity (S/m) and its source.

    Hierarchy (PI: prefer measured prior data, fall back to the model):
      1. McCleskey (2011) Table 3 measurements, where tabulated: 0.5-1.0
         mol/kg and 5-90 C. NaCl via vessel_1bar_check.sigma_prior_nacl, KCl
         by the same method (above). Accuracy assumed ~1 % (see
         plot_sigma_P_main.ANCHOR_UNC).
      2. McCleskey et al. (2012) model (sigmaElectricMcCleskey2012), outside
         that range (e.g. below 5 C). Assumed ~2 %.
    Returns (sigma_Sm, source string).
    """
    inside = (M2011_RANGE_M[0] <= molality <= M2011_RANGE_M[1]
              and M2011_RANGE_T_C[0] <= T_C <= M2011_RANGE_T_C[1])
    if inside and solute == 'NaCl':
        return vbc.sigma_prior_nacl(molality, T_C), 'McCleskey (2011)'
    if inside and solute == 'KCl':
        return _sigma_2011_kcl(molality, T_C), 'McCleskey (2011)'
    return vbc._mc12(T_C, _ions(solute, molality)), 'McCleskey et al. (2012)'


def anchor_info(run_df):
    """Numbers behind the anchor of one run (one row's worth of the factors table)."""
    s0 = first_ambient(run_df)
    ref, src = sigma_ref_1bar(s0.solute, s0.molality, s0.T_C)
    return dict(run=s0.run, solute=s0.solute, molality=s0.molality, T_C=s0.T_C,
                sigma_meas_1bar=s0.sigma_Sm, sigma_ref=ref, ref_source=src,
                factor=ref / s0.sigma_Sm, pct_change=100 * (ref / s0.sigma_Sm - 1))


def anchor_to_1bar(run_df):
    """Scale one run so its first ascending 1-bar step equals the reference.

    factor = sigma_ref_1bar(T of that step) / sigma_measured(that step); every
    sigma (and sigma_lo, sigma_hi) of the run is multiplied by it. This keeps
    the run's MEASURED PRESSURE DEPENDENCE (the ratios sigma(P)/sigma(1 bar)
    are unchanged) and replaces its absolute calibration with prior 1-bar data:
    'trust the composition, calibrate the 1-bar point against prior data'
    (PI, 2026-11-18, applied to Nov 16). The run's own cell constant is thereby
    discarded, and any error in the first 1-bar step carries to the whole
    isotherm. Returns (factor, scaled DataFrame with columns anchor_factor,
    anchor_source).
    """
    info = anchor_info(run_df)
    out = run_df.copy()
    for c in ('sigma_Sm', 'sigma_lo', 'sigma_hi'):
        out[c] = out[c] * info['factor']
    out['anchor_factor'] = info['factor']
    out['anchor_source'] = info['ref_source']
    return info['factor'], out
