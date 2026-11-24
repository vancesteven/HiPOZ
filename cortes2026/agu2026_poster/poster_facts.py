#!/usr/bin/env python3
"""
Quantitative facts for the AGU 2026 poster, computed from the existing CSVs.

Reuses (does not re-derive) the main figure's run selection, exclusions and
anchoring: plot_sigma_P_main.MAIN_RUNS / load_main() and sigma_P_sources.
No sweep fitting, no re-fitting of any circuit; reads the step CSVs only.

Run from anywhere (it changes into the repo root on import, like the other
scripts; PlanetProfile may prompt, so `echo n |` is safe):

    echo n | python cortes2026/agu2026_poster/poster_facts.py

Writes poster_facts.md and poster_facts.csv next to this file.
make_poster_figures.py imports compute() from here, so figures and facts
cannot disagree.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
CORTES = os.path.dirname(HERE)
sys.path.insert(0, CORTES)
import plot_sigma_P_main as pm  # noqa: E402  (imports sigma_P_sources, chdir to repo root)
import sigma_P_sources as src  # noqa: E402
import vessel_1bar_check as vbc  # noqa: E402
import pan2021_nacl as pan  # noqa: E402

P_TARGET = 200.0            # MPa
UD_PRESSURES = (20.0, 190.0)  # window (MPa) for up/down comparison
SOLUTIONS = {'NaCl 0.5': ('NaCl', 0.5), 'NaCl 0.9': ('NaCl', 0.9), 'KCl 0.70': ('KCl', 0.70)}


# ------------------------------------------------------------------ helpers
def solution_of(g):
    for k, (sol, m) in SOLUTIONS.items():
        if g.solute.iloc[0] == sol and abs(g.molality.iloc[0] - m) < 0.05:
            return k
    raise ValueError(g.run.iloc[0])


def interp_at(g, P, col='sigma_Sm', maxgap=60.0, extrap=3.0):
    """Linear interpolation of the ASCENDING steps at pressure P. If P lies up to
    `extrap` MPa beyond the highest compression step (e.g. 199.4 vs 200), the line
    through the last two steps is extended. NaN if the two neighbouring steps are
    more than maxgap MPa apart, or P is further outside the measured range."""
    a = g[~g.descent].sort_values('P_MPa')
    x, y = a.P_MPa.to_numpy(), a[col].to_numpy()
    if P > x.max() and P - x.max() <= extrap and len(x) > 1 and x[-1] - x[-2] <= maxgap:
        return float(y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (P - x[-1]))
    if not (x.min() <= P <= x.max()):
        return np.nan
    i = np.searchsorted(x, P)
    lo, hi = max(i - 1, 0), min(i, len(x) - 1)
    if x[hi] - x[lo] > maxgap:
        return np.nan
    return float(np.interp(P, x, y))


def parabola_max(P, S):
    c = np.polyfit(P, S, 2)
    if c[0] >= 0:
        return np.nan
    return float(-c[1] / (2 * c[0]))


def top_window_max(g, half=100.0):
    """Pressure of the maximum: parabola through the ascending steps within
    +/-half MPa of the highest-sigma step (needs >= 4 points)."""
    a = g[~g.descent]
    ptop = a.P_MPa.iloc[a.sigma_Sm.to_numpy().argmax()]
    w = a[(a.P_MPa >= ptop - half) & (a.P_MPa <= ptop + half)]
    if len(w) < 4:
        return np.nan, ptop
    return parabola_max(w.P_MPa, w.sigma_Sm), ptop


def updown(g, s1):
    """Compression vs decompression at matching pressures, percentage points of
    sigma/sigma(1 bar), plus the return-to-1-bar drift (percent)."""
    d = g[g.descent]
    if d.empty:
        return dict(n_down=0, ud_max_pp=np.nan, ud_mean_pp=np.nan, ud_rms_pp=np.nan,
                    return_1bar_pct=np.nan)
    dd = d[(d.P_MPa > UD_PRESSURES[0]) & (d.P_MPa < UD_PRESSURES[1])]
    diffs = [100 * (r.sigma_Sm - interp_at(g, r.P_MPa, maxgap=130)) / s1 for r in dd.itertuples()]
    diffs = np.array([x for x in diffs if np.isfinite(x)])
    amb = d[d.P_MPa < src.AMBIENT_MAX_MPA]
    ret = 100 * (amb.sigma_Sm.iloc[-1] / s1 - 1) if len(amb) else np.nan
    return dict(n_down=len(d), ud_max_pp=float(np.abs(diffs).max()) if len(diffs) else np.nan,
                ud_mean_pp=float(diffs.mean()) if len(diffs) else np.nan,
                ud_rms_pp=float(np.sqrt((diffs ** 2).mean())) if len(diffs) else np.nan,
                return_1bar_pct=float(ret))


def a_max_P(g):
    return g[~g.descent].P_MPa.max()


def analyse(g):
    """One row of the per-isotherm table from an ANCHORED run (kept steps)."""
    g = g.sort_values('time')
    s0 = src.first_ambient(g)
    s1 = s0.sigma_Sm
    s200 = interp_at(g, P_TARGET)
    lo200, hi200 = interp_at(g, P_TARGET, 'sigma_lo'), interp_at(g, P_TARGET, 'sigma_hi')
    row = dict(run=g.run.iloc[0], solution=solution_of(g), source=g.source.iloc[0],
               date_first=g.time.min().strftime('%Y-%m-%d'), date_last=g.time.max().strftime('%Y-%m-%d'),
               T_C=float(g.T_C.mean()), T_1bar_step_C=float(s0.T_C), P_1bar_step_MPa=float(s0.P_MPa),
               n_steps=len(g), n_down=int(g.descent.sum()),
               anchor_source=g.anchor_source.iloc[0], anchor_factor=float(g.anchor_factor.iloc[0]),
               sigma_1bar_anch=float(s1), sigma_200_anch=s200)
    near = g[~g.descent].iloc[(g[~g.descent].P_MPa - P_TARGET).abs().argsort()[:1]].P_MPa.iloc[0]
    row['nearest_step_to_200_MPa'] = float(near)
    if not np.isfinite(s200):
        row['method_200'] = 'not available: no compression step brackets 200 MPa within 60 MPa'
    elif abs(near - P_TARGET) < 0.005:
        row['method_200'] = 'measured step at 200.0 MPa'
    elif near > P_TARGET or a_max_P(g) > P_TARGET:
        row['method_200'] = ('linear interpolation between the two compression steps bracketing 200 MPa '
                             '(nearest step %.1f MPa)' % near)
    else:
        row['method_200'] = ('linear extrapolation of %.1f MPa to 200 MPa along the last two compression '
                             'steps (highest step %.1f MPa)' % (near, near))
    row['pct_200'] = 100 * (s200 / s1 - 1) if np.isfinite(s200) else np.nan
    # step-range uncertainty (min/max of the sweeps in the step), worst case
    lo1, hi1 = s0.sigma_lo, s0.sigma_hi
    row['pct_200_steprange'] = 0.5 * 100 * (hi200 / lo1 - lo200 / hi1) if np.isfinite(s200) else np.nan
    row.update(updown(g, s1))
    # max / highest pressure
    a = g[~g.descent]
    top = a.iloc[a.sigma_Sm.to_numpy().argmax()]
    P_par, ptop = top_window_max(g)
    Pg, Sg = pm.guide(g)
    row.update(P_max_step_MPa=float(top.P_MPa), pct_at_max_step=100 * (top.sigma_Sm / s1 - 1),
               P_max_parabola_MPa=P_par, P_max_guide_MPa=float(Pg[Sg.argmax()]),
               P_hi_MPa=float(a.P_MPa.max()),
               pct_at_Phi=100 * (a.sigma_Sm.iloc[a.P_MPa.to_numpy().argmax()] / s1 - 1))
    # max bracketed only if the highest-P step is >= 150 MPa above the max step and n >= 9
    row['max_bracketed'] = bool(len(g) >= 9 and a.P_MPa.max() - top.P_MPa >= 150)
    row['pct_at_max_guide'] = 100 * (Sg.max() / s1 - 1)
    return row


# --------------------------------------------------------- literature
def adams_hall_stats():
    ah = src.adams_hall_nacl()
    up = ah[~ah.descent].sort_values('P_MPa')
    ratio = up.sigma_Sm.to_numpy() / up.sigma_Sm.iloc[0]
    pch = PchipInterpolator(up.P_MPa.to_numpy(), ratio)
    P = np.linspace(up.P_MPa.min(), up.P_MPa.max(), 4000)
    ptop = up.P_MPa.iloc[up.sigma_Sm.to_numpy().argmax()]
    w = up[(up.P_MPa >= ptop - 100) & (up.P_MPa <= ptop + 100)]
    return dict(ratio_200=float(pch(200.0)), pct_200=100 * (float(pch(200.0)) - 1),
                P_max_pchip=float(P[pch(P).argmax()]), P_max_parabola=parabola_max(w.P_MPa, w.sigma_Sm),
                n_window=len(w), pct_max=100 * (float(pch(P).max()) - 1),
                P_hi=float(up.P_MPa.max()), pct_hi=100 * (ratio[-1] - 1),
                sigma0=float(up.sigma_Sm.iloc[0]), up=up, ah=ah, pchip=pch)


def pan_sigma(m, T_K, P):
    """pan.sigma_vs_P, padded so that a single pressure does not confuse seafreeze."""
    P = np.atleast_1d(np.asarray(P, float))
    if len(P) == 1:
        return pan.sigma_vs_P(m, T_K, np.concatenate([P, P + 1.0]))[:1]
    return pan.sigma_vs_P(m, T_K, P)


def pan_ratio(m, T_C, P=P_TARGET):
    s = pan.sigma_vs_P(m, T_C + 273.15, np.array([0.1, P]))
    return float(s[1] / s[0]), float(s[0])


def pan_curve(m, T_C, P):
    P = np.asarray(P, float)
    s = pan.sigma_vs_P(m, T_C + 273.15, np.concatenate([[0.1], P]))
    return s[1:] / s[0]


def sigma_ref_vs_T(sol, m, T):
    return src.sigma_ref_1bar(sol, m, T)[0]


def T_equivalent(sol, m, T0, pct):
    """Temperature change at 1 bar (reference conductivity) giving +pct % in sigma."""
    s0 = sigma_ref_vs_T(sol, m, T0)
    f = lambda T: sigma_ref_vs_T(sol, m, T) / s0 - 1 - pct / 100
    T1 = brentq(f, T0 - 0.5, T0 + 30)
    slope = (sigma_ref_vs_T(sol, m, T0 + 0.5) - sigma_ref_vs_T(sol, m, T0 - 0.5)) / s0 * 100
    return T1 - T0, slope


# --------------------------------------------------------------- compute
def compute():
    d, factors = pm.load_main()           # anchored kept steps of MAIN_RUNS
    raw = pd.concat([src.our_steps_2025(False), src.our_steps_2026(False)], ignore_index=True)
    raw = raw[raw.run.isin(pm.MAIN_RUNS)]
    rows, check = [], []
    for run in pm.MAIN_RUNS:
        g = d[d.run == run]
        r = analyse(g)
        # anchoring independence check: same ratio from the UNANCHORED steps
        gr = raw[raw.run == run].copy()
        s1r = src.first_ambient(gr).sigma_Sm
        s200r = interp_at(gr, P_TARGET)
        r['pct_200_unanchored'] = 100 * (s200r / s1r - 1) if np.isfinite(s200r) else np.nan
        r['sigma_1bar_measured'] = float(s1r)
        r['sigma_200_measured'] = s200r
        rows.append(r)
        if np.isfinite(r['pct_200']):
            check.append(abs(r['pct_200'] - r['pct_200_unanchored']))
    tab = pd.DataFrame(rows)
    # Pan 2021 at each isotherm's T and molality
    mol = {'NaCl 0.5': 0.5, 'NaCl 0.9': 0.902, 'KCl 0.70': None}
    pr, pin = [], []
    for r in tab.itertuples():
        m = mol[r.solution]
        if m is None:
            pr.append(np.nan)
            pin.append('')
            continue
        pr.append(100 * (pan_ratio(m, r.T_1bar_step_C)[0] - 1))
        T_K = r.T_1bar_step_C + 273.15
        pin.append('P: outside (%g-%g MPa); T: %s' % (pan.P_RANGE_MPA[0], pan.P_RANGE_MPA[1],
                                                       'inside' if pan.T_RANGE_K[0] <= T_K <= pan.T_RANGE_K[1] else 'outside'))
    tab['pan_pct_200'] = pr
    tab['pan_envelope'] = pin
    return dict(d=d, raw=raw, factors=factors, tab=tab, anchor_check_max_abs=max(check),
                ah=adams_hall_stats())


# ------------------------------------------------------------ fits etc.
def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    c = np.polyfit(x, y, 1)
    res = y - np.polyval(c, x)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n = len(x)
    se = np.sqrt((res ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()) if n > 2 else np.nan
    return dict(slope=float(c[0]), icpt=float(c[1]), r2=float(r2), n=n, slope_se=float(se),
                rms=float(np.sqrt((res ** 2).mean())))



# ------------------------------------------------------------ markdown
def fmt(x, n=1, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    if round(x, n) == 0:
        x = 0.0
    return ('%+.*f' if plus else '%.*f') % (n, x)


def md_table(df, cols, heads, nd):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '---|' * len(heads)]
    for _, r in df.iterrows():
        cells = []
        for c, n in zip(cols, nd):
            v = r[c]
            cells.append(str(v) if n is None else fmt(float(v), n))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def mc12(T_C, sol='NaCl', m=0.902):
    return vbc._mc12(T_C, src._ions(sol, m))


def T_equiv_fn(f, T0, pct):
    s0 = f(T0)
    g = lambda T: f(T) / s0 - 1 - pct / 100
    T1 = brentq(g, T0 - 0.5, T0 + 40)
    slope = (f(T0 + 0.5) - f(T0 - 0.5)) / s0 * 100
    return T1 - T0, slope


def lit_ratio_200(source, c):
    d = pd.read_csv(os.path.join(src.OUT, 'kcl_literature_ratio.csv'))
    d = d[(d.source == source) & np.isclose(d.c, c)].sort_values('P_MPa')
    return 100 * (float(np.interp(P_TARGET, d.P_MPa, d.ratio)) - 1), float(d.P_MPa.max())


def compute_extra(res):
    """Everything beyond the per-isotherm table: fits, T-equivalents, Pan, sensitivities."""
    t = res['tab']
    x = {}
    # (a) T dependence of the 200 MPa increase
    x['fit'] = {}
    for sol in SOLUTIONS:
        s = t[(t.solution == sol) & t.pct_200.notna()]
        x['fit'][sol] = linfit(s.T_C, s.pct_200)
    s = t[(t.solution == 'NaCl 0.5') & t.pct_200.notna() & (abs(t.T_C - 9.85) > 0.5)]
    x['fit_05_no10'] = linfit(s.T_C, s.pct_200)
    # (b) temperature equivalent, 0.902 mol/kg NaCl
    n9 = t[(t.solution == 'NaCl 0.9') & t.pct_200.notna()].copy()
    eq = []
    for r in n9.itertuples():
        dT12, sl12 = T_equiv_fn(mc12, r.T_1bar_step_C, r.pct_200)
        if r.T_1bar_step_C >= 5:
            f11 = lambda T: src.sigma_ref_1bar('NaCl', 0.902, T)[0]
            dT11, sl11 = T_equiv_fn(f11, r.T_1bar_step_C, r.pct_200)
        else:
            dT11 = sl11 = np.nan
        eq.append(dict(run=r.run, T=r.T_1bar_step_C, pct=r.pct_200, dT_mc12=dT12, slope_mc12=sl12,
                       dT_mc11=dT11, slope_mc11=sl11))
    x['equiv'] = pd.DataFrame(eq)
    # same, from OUR OWN raw 1-bar values (one K for Nov 17-19): slope between neighbouring isotherms
    raw = t[t.run.str.contains('Nov 17|Nov 18|Nov 19')].sort_values('T_1bar_step_C')
    sl = []
    for a, b in zip(raw.itertuples(), list(raw.itertuples())[1:]):
        sl.append((a.T_1bar_step_C, b.T_1bar_step_C,
                   100 * (b.sigma_1bar_measured / a.sigma_1bar_measured - 1) / (b.T_1bar_step_C - a.T_1bar_step_C)
                   / (0.5 * (b.sigma_1bar_measured / a.sigma_1bar_measured + 1))))
    x['raw_slopes'] = sl
    # (c) adams & hall vs ours
    ah = res['ah']
    x['ah_at400'] = 100 * (float(ah['pchip'](400.0)) - 1)
    d = res['d']
    c400 = []
    for run, g in d[d.run.str.contains('NaCl 0.5')].groupby('run', sort=False):
        s1 = src.first_ambient(g).sigma_Sm
        q = g[(g.P_MPa > 390) & (g.P_MPa < 410)]
        c400.append((run, g.T_C.mean(), 100 * (q.sigma_Sm.mean() / s1 - 1) if len(q) else np.nan))
    x['ours_at400'] = c400
    # ~600 MPa behaviour relative to the maximum
    r600 = []
    for run, g in d[d.run.str.contains('NaCl 0.5')].groupby('run', sort=False):
        s1 = src.first_ambient(g).sigma_Sm
        q = g[(g.P_MPa > 590) & (g.P_MPa < 610)]
        smax = g.sigma_Sm.max()
        r600.append((run, g.T_C.mean(), float(q.P_MPa.iloc[0]) if len(q) else np.nan,
                     100 * (q.sigma_Sm.iloc[0] / s1 - 1) if len(q) else np.nan,
                     q.sigma_Sm.iloc[0] / smax if len(q) else np.nan,
                     100 * (smax / s1 - 1)))
    x['at600'] = r600
    # (d) Pan
    pr = []
    for r in t[t.solution != 'KCl 0.70'].itertuples():
        m = 0.5 if r.solution == 'NaCl 0.5' else 0.902
        ratio, s_pan1 = pan_ratio(m, r.T_1bar_step_C)
        r212 = pan_ratio(m, r.T_1bar_step_C, 212.0)[0]
        ref1 = src.sigma_ref_1bar('NaCl', m, r.T_1bar_step_C)[0]
        pr.append(dict(run=r.run, solution=r.solution, T=r.T_1bar_step_C, ours=r.pct_200,
                       pan=100 * (ratio - 1), pan212=100 * (r212 - 1), pan_1bar_vs_ref=100 * (s_pan1 / ref1 - 1),
                       abs200=(100 * (r.sigma_200_anch / (s_pan1 * ratio) - 1) if np.isfinite(r.sigma_200_anch) else np.nan),
                       T_in=bool(pan.T_RANGE_K[0] <= r.T_1bar_step_C + 273.15 <= pan.T_RANGE_K[1])))
    x['pan'] = pd.DataFrame(pr)
    # Pan inside its envelope, 0.5 mol/kg: absolute sigma at our steps with P >= 212 MPa and T inside
    cmp_ = []
    for run, g in d[d.run.str.contains('NaCl 0.5')].groupby('run', sort=False):
        T_K = g.T_C.mean() + 273.15
        if not (pan.T_RANGE_K[0] <= T_K <= pan.T_RANGE_K[1]):
            continue
        q = g[g.P_MPa >= pan.P_RANGE_MPA[0]]
        sp = pan_sigma(0.5, T_K, q.P_MPa.to_numpy())
        cmp_.append((run, g.T_C.mean(), len(q), float((100 * (q.sigma_Sm.to_numpy() / sp - 1)).mean()),
                     float((100 * (q.sigma_Sm.to_numpy() / sp - 1)).min()),
                     float((100 * (q.sigma_Sm.to_numpy() / sp - 1)).max())))
    x['pan_inside'] = cmp_
    pP = np.linspace(0.1, 1700, 3000)
    x['pan_pmax'] = {T: float(pP[pan.sigma_vs_P(0.902, T + 273.15, pP).argmax()]) for T in (2, 25)}
    # (e) temperature range, Aug 2025 (all steps incl. excluded, and kept)
    a25 = pd.read_csv(src.STEPS_2025_CSV, keep_default_na=False)
    kept25 = a25[a25.excluded == '']
    x['T25'] = (a25.T_C.min(), a25.T_C.max(), kept25.T_C.min(), kept25.T_C.max())
    # (f) KCl vs NaCl 0.9 at matched T
    kc = t[(t.solution == 'KCl 0.70') & t.pct_200.notna()]
    f9 = x['fit']['NaCl 0.9']
    x['kcl_vs_nacl'] = [(r.T_C, r.pct_200, f9['icpt'] + f9['slope'] * r.T_C,
                         (r.pct_200 - (f9['icpt'] + f9['slope'] * r.T_C))) for r in kc.itertuples()]
    x['kcl_vs_nacl_pairs'] = []
    n9t = t[(t.solution == 'NaCl 0.9') & t.pct_200.notna()]
    for r in kc.itertuples():
        j = (n9t.T_C - r.T_C).abs().idxmin()
        x['kcl_vs_nacl_pairs'].append((r.T_C, r.pct_200, n9t.loc[j, 'T_C'], n9t.loc[j, 'pct_200']))
    x['lit'] = {}
    for srcname, c in (('Fisher and Fox 1979', 0.02), ('Fisher and Fox 1979', 0.002), ('Fisher 1962', 0.02),
                       ('Fisher 1962', 0.01), ('Adams and Hall 1931', 0.05022)):
        x['lit'][(srcname, c)] = lit_ratio_200(srcname, c)
    # (g) cell constants and 1-bar offsets of Nov 17-19 for alternative K
    k = pd.read_csv(src.STEPS_2026_CSV)
    Kused = k[k.run.str.contains('Nov 17|Nov 18|Nov 19')].K_cell.unique()
    x['K_nov17'] = float(Kused[0])
    x['K_kcl_pre'] = float(k[k.run.str.contains('KCl 8 S/m std, Nov')].K_cell.unique()[0])
    x['K_nov16'] = float(k[k.run == 'NaCl 0.9 m, Nov 16'].K_cell.iloc[0])
    x['K_oct'] = float(k[k.run.str.contains('NaCl 0.9 m, Oct 7-8')].K_cell.iloc[0])
    off = []
    for r in t[t.run.str.contains('Nov 17|Nov 18|Nov 19')].itertuples():
        ref = src.sigma_ref_1bar('NaCl', 0.902, r.T_1bar_step_C)
        row = dict(run=r.run, T=r.T_1bar_step_C, ref_src=ref[1])
        for kk in K_ALT:
            row['off_%s' % kk] = 100 * (r.sigma_1bar_measured * K_ALT[kk] / x['K_nov17'] / ref[0] - 1)
        off.append(row)
    x['offsets'] = pd.DataFrame(off)
    # hypothetical series-resistance reading of the anchor factors (Nov 17-19, K = K_nov17 assumed right)
    sr = []
    dd = res['d']
    for r in t[t.run.str.contains('Nov 17|Nov 18|Nov 19')].itertuples():
        f = r.anchor_factor
        g = dd[dd.run == r.run]
        s1m = r.sigma_1bar_measured                      # measured (unanchored) 1-bar
        s200m = r.sigma_200_measured
        rs_over_K = (1 / s1m) * (1 - 1 / f)               # Rs/K, from R_meas = R_true + Rs at 1 bar
        s200_true = 1 / (1 / s200m - rs_over_K)
        s1_true = 1 / (1 / s1m - rs_over_K)
        sr.append((r.run, r.T_1bar_step_C, r.pct_200, 100 * (s200_true / s1_true - 1)))
    x['series'] = sr
    # alternative anchor for the 16.2 C isotherm: closing 1-bar step
    alt = []
    for r in t[t.run.str.contains('Nov 17|Nov 18|Nov 19|Nov 16|Oct 7')].itertuples():
        g = dd[dd.run == r.run].sort_values('time')
        close = g[g.descent & (g.P_MPa < src.AMBIENT_MAX_MPA)]
        if len(close):
            alt.append((r.run, r.pct_200, 100 * (r.sigma_200_anch / close.sigma_Sm.iloc[0] - 1)
                        if np.isfinite(r.sigma_200_anch) else np.nan, close.P_MPa.iloc[0]))
    x['alt_anchor'] = alt
    return x


K_ALT = {'Diaz config K 116.84': 116.84, 'Nov 19 standard K 115.21': 115.21}


def write(res):
    t = res['tab'].copy()
    x = compute_extra(res)
    ah = res['ah']
    L = []
    P = L.append
    t['dT_equiv'] = np.nan
    P('# AGU 2026 poster facts: sigma(P) of aqueous NaCl and KCl')
    P('')
    P('Generated by `cortes2026/agu2026_poster/poster_facts.py` (re-run it; every number below is computed from the')
    P('step CSVs in `cortes2026/nacl_sigma_P/` with the main figure\'s run selection, exclusions and anchoring).')
    P('Nothing here was re-fitted. Items marked QUOTED are constants copied from repository documents, not computed.')
    P('')
    P('Conventions. "1 bar" = the run\'s first kept ascending ambient step (P < 3 MPa; its actual pressure is in the CSV, '
      '0.06-1.7 MPa). sigma values are ANCHORED (1-bar step set to McCleskey 2011/2012), as in the main figure. '
      '"sigma(200 MPa)" = linear interpolation between the two compression steps that bracket 200 MPa, or, when the '
      'highest step is 0.0-3 MPa below 200, the line through the last two compression steps extended to 200 (method '
      'column in the CSV). "Up/down" = difference of sigma/sigma(1 bar) between decompression and compression at the '
      'same pressure (decompression pressure; compression interpolated), 20-190 MPa, percentage points of the 1-bar '
      'value. "Step range" = half-width of the pct increase using the min/max sweep of each step.')
    P('')
    P('## Per-isotherm table')
    P('')
    P('Anchoring independence check: the percent increase to 200 MPa was recomputed from the UNANCHORED steps; the '
      'largest absolute difference over all isotherms is %.1e percentage points, so the ratio is independent of the '
      'anchoring (anchoring multiplies a whole run by one factor).' % res['anchor_check_max_abs'])
    P('')
    t['dates'] = np.where(t.date_first == t.date_last, t.date_first, t.date_first + ' to ' + t.date_last)
    t['Tn'] = t['T_C']
    P(md_table(t, ['solution', 'dates', 'T_C', 'sigma_1bar_anch', 'sigma_200_anch', 'pct_200', 'pct_200_steprange',
                   'ud_max_pp', 'return_1bar_pct', 'anchor_factor'],
               ['solution', 'date(s)', 'T (C)', 'sigma 1 bar (S/m)', 'sigma 200 MPa (S/m)', '% 1 bar to 200 MPa',
                '+/- step range (pp)', 'up/down max (pp)', 'return to 1 bar (%)', 'anchor factor'],
               [None, None, 2, 3, 3, 2, 2, 2, 2, 4]))
    P('')
    P('Run keys in the same order: ' + '; '.join('%s' % r for r in t.run) + '.')
    P('')
    P('0.5 mol/kg NaCl only (maximum of sigma(P)); the pressure of the maximum is given three ways because the steps are '
      '50 MPa apart: the highest-sigma compression step; a parabola through the compression steps within +/-100 MPa of '
      'it (needs >= 4 points); and the cubic guide polynomial of the main figure. "% at max" uses the highest-sigma step.')
    P('')
    s5 = t[t.solution == 'NaCl 0.5']
    P(md_table(s5, ['T_C', 'P_max_step_MPa', 'P_max_parabola_MPa', 'P_max_guide_MPa', 'pct_at_max_step', 'P_hi_MPa',
                    'pct_at_Phi', 'n_down'],
               ['T (C)', 'P max, step (MPa)', 'P max, parabola (MPa)', 'P max, guide (MPa)', '% at max vs 1 bar',
                'highest P (MPa)', '% at highest P vs 1 bar', 'kept decompression steps'],
               [2, 0, 0, 0, 1, 0, 1, 0]))
    P('')
    P('Aug 20 20 C has compression steps only at 0.1, 48, 100 and 607 MPa (the 607 MPa step belongs to it per the '
      'sigma_P_sources docstring), so it has no 200 MPa value and no maximum; the maximum is not bracketed there. '
      'No decompression step of any Aug 2025 isotherm is kept (the hasty Aug 19/20 decompression steps are excluded), '
      'so there is no up/down repeatability for 0.5 mol/kg: only the sweep-to-sweep step range is available.')
    P('')
    P('Up/down repeatability (2026 isotherms): max |down - up| over 20-190 MPa = %s to %s percentage points of the '
      '1-bar value; return-to-1-bar drift (last decompression ambient step vs first) = %s to %s %%.' % (
          fmt(t.ud_max_pp.min(), 2), fmt(t.ud_max_pp.max(), 2), fmt(t.return_1bar_pct.min(), 2),
          fmt(t.return_1bar_pct.max(), 2)))
    P('')
    # ---------------- a
    P('## a. Temperature dependence of the percent increase to 200 MPa')
    P('')
    P('Method: ordinary least-squares line of pct_200 vs isotherm mean T for each solution (n, slope +/- 1 se, r^2, rms '
      'residual); values quoted are the measured isotherms at the ends of the T range.')
    P('')
    for sol in SOLUTIONS:
        f = x['fit'][sol]
        s = t[(t.solution == sol) & t.pct_200.notna()].sort_values('T_C')
        lo, hi = s.iloc[0], s.iloc[-1]
        P('- **%s mol/kg**: %s%% at %s C (coldest) to %s%% at %s C (warmest) over %d isotherms; linear fit slope %s +/- %s pp per C '
          '(r^2 = %s, rms residual %s pp); fit value at 25 C = %s%%.' % (
              sol.split()[0] + ' ' + sol.split()[1], fmt(lo.pct_200), fmt(lo.T_C), fmt(hi.pct_200), fmt(hi.T_C), f['n'],
              fmt(f['slope'], 3), fmt(f['slope_se'], 3), fmt(f['r2'], 2), fmt(f['rms'], 2),
              fmt(f['icpt'] + 25 * f['slope'])))
    f = x['fit_05_no10']
    P('  - NaCl 0.5 without the 9.85 C isotherm (its 1-bar step looks unsettled, see h): slope %s +/- %s pp per C, r^2 = %s, '
      'n = %d, fit value at 25 C = %s%%.' % (fmt(f['slope'], 3), fmt(f['slope_se'], 3), fmt(f['r2'], 2), f['n'],
                                              fmt(f['icpt'] + 25 * f['slope'])))
    n9 = t[(t.solution == 'NaCl 0.9') & t.pct_200.notna()].sort_values('T_C')
    P('')
    kk = t[(t.solution == 'KCl 0.70')].sort_values('T_C')
    mono9 = bool((np.diff(n9.pct_200.to_numpy()) <= 0).all())
    viol = [(a.run.replace('NaCl 0.9 m, ', ''), a.T_C, a.pct_200, b.run.replace('NaCl 0.9 m, ', ''), b.T_C, b.pct_200)
            for a, b in zip(list(n9.itertuples()), list(n9.itertuples())[1:]) if b.pct_200 > a.pct_200]
    monok = bool((np.diff(kk.pct_200.to_numpy()) <= 0).all())
    o78 = t[t.run.str.contains('Oct 7')].iloc[0]
    n16 = t[t.run.str.contains('Nov 16')].iloc[0]
    P('Plain statement (0.9 mol/kg NaCl, the headline set, %d isotherms): "%s%% at %s C rising to %s%% at %s C" '
      '(Oct 7-8 isotherm at %s C: %s%%; Nov 16 at %s C: %s%%). Percent increase %s monotonically with falling T across all '
      '%d isotherms (rms residual of the straight line %s pp). KCl 0.70 mol/kg (%d isotherms): %s%% at %s C rising to %s%% '
      'at %s C (%s monotonic in T; rms residual %s pp).' % (
          len(n9), fmt(n9.pct_200.iloc[-1]), fmt(n9.T_C.iloc[-1]), fmt(n9.pct_200.iloc[0]), fmt(n9.T_C.iloc[0]),
          fmt(o78.T_C), fmt(o78.pct_200), fmt(n16.T_C), fmt(n16.pct_200),
          'rises' if mono9 else 'does NOT rise', len(n9), fmt(x['fit']['NaCl 0.9']['rms'], 2),
          len(kk), fmt(kk.pct_200.iloc[-1]), fmt(kk.T_C.iloc[-1]), fmt(kk.pct_200.iloc[0]), fmt(kk.T_C.iloc[0]),
          'is' if monok else 'is not', fmt(x['fit']['KCl 0.70']['rms'], 2)))
    if viol:
        P('Exception to strict monotonicity (0.9 mol/kg): ' + '; '.join(
            '%s at %s C = %s%% is below %s at %s C = %s%%' % (a[0], fmt(a[1]), fmt(a[2]), b[0], fmt(b[1]), fmt(b[2]))
            for a, b in [(v[:3], v[3:]) for v in viol]) + ' (the 16.2 C isotherm\'s anchor step is the one flagged as possibly '
          'unsettled, see g).')
    P('Headline numbers: 0.9 mol/kg NaCl %s%% at %s C to %s%% at %s C; KCl 0.70 %s%% at %s C to %s%% at %s C; '
      '0.5 mol/kg %s%% at %s C to %s%% at %s C (see the 9.85 C caveat).' % (
          fmt(t[t.run.str.contains('Nov 17, 24.8')].pct_200.iloc[0]), fmt(t[t.run.str.contains('Nov 17, 24.8')].T_C.iloc[0]),
          fmt(n9.pct_200.iloc[0]), fmt(n9.T_C.iloc[0]), fmt(kk.pct_200.iloc[-1]), fmt(kk.T_C.iloc[-1]),
          fmt(kk.pct_200.iloc[0]), fmt(kk.T_C.iloc[0]), fmt(t[(t.solution == 'NaCl 0.5') & t.pct_200.notna()].sort_values('T_C').pct_200.iloc[-1]),
          fmt(t[(t.solution == 'NaCl 0.5') & t.pct_200.notna()].sort_values('T_C').T_C.iloc[-1]),
          fmt(t[(t.solution == 'NaCl 0.5') & t.pct_200.notna()].sort_values('T_C').pct_200.iloc[0]),
          fmt(t[(t.solution == 'NaCl 0.5') & t.pct_200.notna()].sort_values('T_C').T_C.iloc[0])))
    P('')
    P('KCl per isotherm (T in C: % to 200 MPa): ' + '; '.join('%s: %s' % (fmt(r.T_C), fmt(r.pct_200)) for r in kk.itertuples()) + '.')
    P('')
    # ---------------- b
    P('## b. Temperature equivalent of the pressure effect, 0.902 mol/kg NaCl')
    P('')
    P('Method: dsigma/dT at 1 bar from the 1-bar reference conductivity (McCleskey et al. 2012 model, smooth in T; '
      'McCleskey 2011 spline as cross-check where T >= 5 C). dT = the temperature change at 1 bar that gives the same '
      'relative sigma change as 0 to 200 MPa at the isotherm temperature (solved exactly, not linearised); slope = '
      'dln(sigma)/dT in %/K at that T.')
    P('')
    e = x['equiv']
    P(md_table(e, ['T', 'pct', 'slope_mc12', 'dT_mc12', 'slope_mc11', 'dT_mc11'],
               ['T (C)', 'pct to 200 MPa', 'slope MC12 (%/K)', 'dT MC12 (K)', 'slope MC11 (%/K)', 'dT MC11 (K)'],
               [2, 2, 2, 2, 2, 2]))
    P('')
    e25 = e.iloc[(e['T'] - 24.9).abs().argmin()]
    e2 = e.iloc[(e['T'] - 1.7).abs().argmin()]
    e7 = e.iloc[(e['T'] - 7.0).abs().argmin()]
    P('Answer: at %s C the pressure effect (%s%%) equals a warming of %s K (MC12; %s K from MC11); at %s C (%s%%) it equals '
      '%s K (MC12; MC11 not defined below 5 C). Sensitivity from OUR OWN raw 1-bar values (single K for Nov 17-19, '
      'neighbouring isotherms): %s. These raw slopes include the post-repair offset that grows on cooling (see g), so '
      'the reference-based numbers are preferred.' % (
          fmt(e25['T']), fmt(e25.pct), fmt(e25.dT_mc12), fmt(e25.dT_mc11), fmt(e2['T']), fmt(e2.pct), fmt(e2.dT_mc12),
          '; '.join('%s-%s C: %s %%/K' % (fmt(a), fmt(b), fmt(s, 2)) for a, b, s in x['raw_slopes'])))
    P('')
    draft = e7
    P('Draft poster sentence "nearly 10%% (equivalent to about 1.5 C temperature change)": the data give %s%% at %s C '
      '(the isotherm closest to 10%%), whose equivalent is %s K, and 1.5 K at that temperature corresponds to only '
      '%s%% in sigma (slope %s %%/K x 1.5 K). The 10%% is supported only for the cold isotherms (%s%% at %s C, %s%% at %s C); '
      'the "about 1.5 C" is NOT supported by any isotherm: the equivalent is %s to %s K across the 0.9 mol/kg isotherms.' % (
          fmt(draft.pct), fmt(draft['T']), fmt(draft.dT_mc12), fmt(1.5 * draft.slope_mc12), fmt(draft.slope_mc12, 2),
          fmt(e2.pct), fmt(e2['T']), fmt(draft.pct), fmt(draft['T']), fmt(e.dT_mc12.min()), fmt(e.dT_mc12.max())))
    P('')
    # ---------------- c
    P('## c. Adams and Hall (1931) at 25 C, NaCl 2.865 wt%% (%s mol/kg)' % fmt(src.ADAMS_NACL_MOLAL, 3))
    P('')
    P('Method: their R/R0 table (ascending points only), sigma ratio = R0/R, PCHIP through the points; maximum from the '
      'PCHIP grid and from a parabola through the points within +/-100 MPa of the highest-sigma point (%d points). '
      'Percent figures are relative to their 1 bar (transcription pending independent check, per the CSV header).' % ah['n_window'])
    P('')
    P('- Adams and Hall: +%s%% at 200 MPa; maximum at %s MPa (PCHIP) / %s MPa (parabola), +%s%% at the maximum; '
      '+%s%% at %s MPa (their highest pressure); +%s%% at 400 MPa.' % (
          fmt(ah['pct_200']), fmt(ah['P_max_pchip'], 0), fmt(ah['P_max_parabola'], 0), fmt(ah['pct_max']),
          fmt(ah['pct_hi']), fmt(ah['P_hi'], 0), fmt(x['ah_at400'])))
    s5k = t[(t.solution == 'NaCl 0.5') & (abs(t.T_C - 20) < 2) & t.pct_200.notna()]
    P('- Ours at the nearest temperatures (0.5 mol/kg, %s-%s C, i.e. %s-%s K below 25 C; the Aug 20 19.9 C isotherm has no 200 MPa step): ' % (
        fmt(s5k.T_C.min()), fmt(s5k.T_C.max()), fmt(25 - s5k.T_C.max()), fmt(25 - s5k.T_C.min())) + '; '.join(
        '%s C: +%s%% at 200 MPa, maximum at %s MPa (highest step) / %s (parabola) / %s (guide)' % (
            fmt(r.T_C), fmt(r.pct_200), fmt(r.P_max_step_MPa, 0), fmt(r.P_max_parabola_MPa, 0), fmt(r.P_max_guide_MPa, 0))
        for r in s5k.itertuples()) + '.')
    o400 = [(T, v) for _, T, v in x['ours_at400'] if np.isfinite(v)]
    P('- At 400 MPa (steps within 390-410 MPa) ours vs theirs: ' + '; '.join(
        '%s C: +%s%%' % (fmt(T), fmt(v)) for T, v in o400) + '; theirs +%s%%.' % fmt(x['ah_at400']))
    f05 = x['fit_05_no10']
    P('- Our 0.5 mol/kg increase extrapolated to 25 C by the linear fit (excluding 9.85 C): %s%% (for comparison '
      'only; no 0.5 mol/kg isotherm was run at 25 C). A 5 K temperature difference at that fit slope (%s pp per C) is %s pp.' % (
          fmt(f05['icpt'] + 25 * f05['slope']), fmt(f05['slope'], 2), fmt(abs(5 * f05['slope']))))
    r_ = s5k.pct_200
    o20 = [v for T, v in o400 if abs(T - 20) < 2]
    P('')
    P('Does our result "confirm" theirs? Position of the maximum: consistent. Ours (~20 C) %s-%s MPa by highest step or '
      'parabola, %s-%s MPa by guide polynomial (steps are 50 MPa apart); theirs %s MPa (PCHIP) to %s MPa (parabola). '
      'Magnitude of the increase: not in agreement. Our ~20 C isotherms give +%s%% to +%s%% at 200 MPa against their +%s%% at 25 C '
      '(%s to %s times larger); at 400 MPa +%s%% to +%s%% against +%s%%. The 5 K temperature difference is worth about %s pp '
      '(see above), not the %s pp gap. So "agreement in percent" is NOT supported; only the position of the maximum is. See h.' % (
          fmt(s5k.P_max_step_MPa.min(), 0), fmt(s5k.P_max_parabola_MPa.max(), 0), fmt(s5k.P_max_guide_MPa.min(), 0),
          fmt(s5k.P_max_guide_MPa.max(), 0), fmt(ah['P_max_pchip'], 0), fmt(ah['P_max_parabola'], 0), fmt(r_.min()), fmt(r_.max()),
          fmt(ah['pct_200']), fmt(r_.min() / ah['pct_200'], 1), fmt(r_.max() / ah['pct_200'], 1), fmt(min(o20)),
          fmt(max(o20)), fmt(x['ah_at400']), fmt(abs(5 * f05['slope'])), fmt(r_.min() - ah['pct_200'])))
    P('')
    # ---------------- d
    P('## d. Pan et al. (2021) model')
    P('')
    Tout = t[(t.solution == 'NaCl 0.9') & ((t.T_1bar_step_C + 273.15) > pan.T_RANGE_K[1])]
    P('Validity (pan2021_nacl.py docstring/constants): experiments %g-%g MPa, %g-%g K (%s to %s C), 1/5/10 wt%% NaCl; '
      'the module warns above %g mol/kg. 0-200 MPa is therefore entirely OUTSIDE the pressure range of their data '
      '(lowest data pressure %g MPa); the module docstring states that below ~212 MPa it "misses the conductivity maximum". '
      'Our 0.5 mol/kg (2.9 wt%%) and 0.9 mol/kg (5 wt%%) are inside their concentration range. Our isotherms warmer than their '
      'T limit (295 K = %s C), i.e. also T-extrapolated: 0.9 mol/kg at %s C; 0.5 mol/kg at %s C.' % (
          pan.P_RANGE_MPA[0], pan.P_RANGE_MPA[1], pan.T_RANGE_K[0], pan.T_RANGE_K[1], fmt(pan.T_RANGE_K[0] - 273.15),
          fmt(pan.T_RANGE_K[1] - 273.15), pan.MOLAL_MAX, pan.P_RANGE_MPA[0], fmt(pan.T_RANGE_K[1] - 273.15),
          ', '.join(fmt(v) for v in sorted(Tout.T_1bar_step_C)),
          ', '.join(fmt(v) for v in t[(t.solution == 'NaCl 0.5') & ((t.T_1bar_step_C + 273.15) > pan.T_RANGE_K[1])].T_1bar_step_C)))
    P('')
    P('Method: pan.sigma_vs_P(m, T, [0.1, 200 MPa]) at the same molality (0.5 or 0.902) and the 1-bar-step temperature of '
      'each isotherm; percent change = sigma(200)/sigma(0.1 MPa) - 1. "Pan 212" is the same at the edge of their data '
      '(212 MPa). The last column compares absolute sigma(200 MPa) (ours anchored, Pan absolute).')
    P('')
    pp = x['pan'].copy()
    pp['diff'] = pp.ours - pp.pan
    P(md_table(pp, ['solution', 'T', 'ours', 'pan', 'diff', 'pan212', 'pan_1bar_vs_ref', 'abs200'],
               ['solution', 'T (C)', 'ours % to 200', 'Pan % to 200', 'ours - Pan (pp)', 'Pan % to 212',
                'Pan sigma(0.1 MPa) vs 1-bar reference (%)', 'ours sigma(200)/Pan sigma(200) - 1 (%)'],
               [None, 2, 2, 2, 2, 2, 1, 1]))
    P('')
    p9 = pp[(pp.solution == 'NaCl 0.9') & pp.ours.notna()]
    p25 = p9.iloc[(p9['T'] - 24.9).abs().argmin()]
    pmin = p9.iloc[p9['T'].argmin()]
    P('Numbers: for 0.9 mol/kg the model predicts %s%% to %s%% between 1 bar and 200 MPa (more negative with cooling), where '
      'we measure +%s%% to +%s%% (larger with cooling); ours minus Pan grows from %s pp (%s C) to %s pp (%s C). At %s C ours '
      'exceeds Pan by %s pp; at the coldest isotherm (%s C) the model gives %s%% against our +%s%%. The model\'s own pressure '
      'of maximum over 0.1-1700 MPa for 0.902 mol/kg is %s MPa at 2 C and %s MPa at 25 C.' % (
          fmt(p9.pan.min(), 1, True), fmt(p9.pan.max(), 1, True), fmt(p9.ours.min()), fmt(p9.ours.max()),
          fmt(p9['diff'].min()), fmt(p9.iloc[p9['diff'].argmin()]['T']), fmt(p9['diff'].max()),
          fmt(p9.iloc[p9['diff'].argmax()]['T']), fmt(p25['T']), fmt(p25['diff'], 1), fmt(pmin['T']),
          fmt(pmin.pan, 1, True), fmt(pmin.ours), fmt(x['pan_pmax'][2], 0), fmt(x['pan_pmax'][25], 0)))
    P('')
    P('Inside their pressure envelope (0.5 mol/kg, our steps with P >= 212 MPa, T within 233-295 K): ours/Pan - 1 for absolute '
      'sigma, mean [min, max]: ' + '; '.join(
          '%s C: %s%% [%s, %s] (n=%d)' % (fmt(T), fmt(m, 1, True), fmt(lo, 1, True), fmt(hi, 1, True), n)
          for _, T, n, m, lo, hi in x['pan_inside']) + '. Our sigma is above the model by this much even where the model is fitted; '
      'see h.')
    P('')
    P('Does the numbers support "contradicts extrapolations from above 200 MPa by Pan et al. (2021)"? The direction is supported: '
      'the Pan regression run below its 212 MPa lower bound predicts a change of %s%% to %s%% over 0-200 MPa for 0.9 mol/kg '
      '(a decrease in the cold), while we measure +%s%% to +%s%% (an increase, larger in the cold). Caveats: it is an '
      'extrapolation of a regression to data at >= 212 MPa that the module itself says misses the low-pressure rise, so '
      '"contradict" overstates it; the model\'s own 1-bar value is %s%% to %s%% from the 1-bar reference for 0.9 mol/kg, i.e. '
      'it is not constrained at low pressure. Suggested wording: "Between 1 bar and 200 MPa, conductivity of 0.9 mol/kg NaCl '
      'rose by %s%% at %s C to %s%% at %s C; the Pan et al. (2021) regression, fitted at 212-1713 MPa, extrapolates to %s%% '
      '(%s C) to %s%% (%s C) over the same range." (the figures are the computed values above, rounded).' % (
          fmt(p9.pan.min(), 1, True), fmt(p9.pan.max(), 1, True), fmt(p9.ours.min()), fmt(p9.ours.max()),
          fmt(p9.pan_1bar_vs_ref.min(), 1, True), fmt(p9.pan_1bar_vs_ref.max(), 1, True),
          fmt(p9.ours.min(), 0), fmt(p9.iloc[p9.ours.argmin()]['T']), fmt(p9.ours.max(), 0), fmt(p9.iloc[p9.ours.argmax()]['T']),
          fmt(p9.pan.max(), 0, True), fmt(p9.iloc[p9.pan.argmax()]['T']), fmt(p9.pan.min(), 0, True),
          fmt(p9.iloc[p9.pan.argmin()]['T'])))
    P('')
    # ---------------- e
    P('## e. "Pressure trends relatively unaffected by temperature in the range measured"')
    P('')
    T25 = x['T25']
    P('Actual temperature range of the Aug 2025 0.5 mol/kg data: kept steps %s-%s C (%s-%s K); all steps including excluded '
      'ones %s-%s C (%s-%s K). The abstract\'s "274-318 K" overstates the upper end by %s K.' % (
          fmt(T25[2], 2), fmt(T25[3], 2), fmt(T25[2] + 273.15), fmt(T25[3] + 273.15), fmt(T25[0], 2), fmt(T25[1], 2),
          fmt(T25[0] + 273.15), fmt(T25[1] + 273.15), fmt(318 - (T25[3] + 273.15))))
    P('')
    s5b = t[(t.solution == 'NaCl 0.5') & t.max_bracketed]
    s5p = t[(t.solution == 'NaCl 0.5') & t.pct_200.notna()].sort_values('T_C')
    s5q = s5p[abs(s5p.T_C - 9.85) > 0.5]
    br = set(s5b.T_C.round(2))
    P('Numbers (0.5 mol/kg): the 200 MPa increase is ' + ', '.join(
        '%s%% at %s C' % (fmt(r.pct_200), fmt(r.T_C)) for r in s5p.itertuples()) +
      '; largest/smallest = %s (%s excluding the suspect 9.85 C isotherm). Pressure of the maximum (isotherms whose maximum is '
      'bracketed: %s): %s-%s MPa by highest step, %s-%s MPa by parabola (where defined), %s-%s MPa by guide polynomial. '
      'sigma at ~600 MPa as a fraction of the isotherm\'s own maximum: ' % (
          fmt(s5p.pct_200.max() / s5p.pct_200.min(), 1), fmt(s5q.pct_200.max() / s5q.pct_200.min(), 1),
          ', '.join(fmt(v) + ' C' for v in sorted(br)),
          fmt(s5b.P_max_step_MPa.min(), 0), fmt(s5b.P_max_step_MPa.max(), 0), fmt(s5b.P_max_parabola_MPa.min(), 0),
          fmt(s5b.P_max_parabola_MPa.max(), 0), fmt(s5b.P_max_guide_MPa.min(), 0), fmt(s5b.P_max_guide_MPa.max(), 0)) +
      ', '.join('%s (%s C)' % (fmt(f, 2), fmt(T)) for _, T, Pq, pc, f, pm_ in x['at600']
                if np.isfinite(f) and round(T, 2) in br) + '. The change at ~600 MPa relative to 1 bar is ' + ', '.join(
          '%s%% (%s C)' % (fmt(pc, 1, True), fmt(T)) for _, T, Pq, pc, f, pm_ in x['at600'] if np.isfinite(pc)) + '.')
    P('')
    P('Verdict (numbers, not interpretation): the position of the maximum and the post-maximum decline are similar at all '
      'temperatures within the 50 MPa step spacing (maxima %s-%s MPa by parabola/guide). The size of the increase is not '
      'temperature-independent: %s%% to %s%% at 200 MPa (slope %s +/- %s pp per C without the 9.85 C isotherm), and %s%% to %s%% '
      'for the 0.9 mol/kg isotherms (slope %s pp per C) and %s%% to %s%% for KCl 0.70. "Relatively unaffected" is supported for '
      'the pressure of the maximum, not for the magnitude of the pressure effect.' % (
          fmt(s5b.P_max_parabola_MPa.min(), 0), fmt(s5b.P_max_guide_MPa.max(), 0), fmt(s5q.pct_200.min()),
          fmt(s5q.pct_200.max()), fmt(f05['slope'], 2), fmt(f05['slope_se'], 2), fmt(n9.pct_200.min()), fmt(n9.pct_200.max()),
          fmt(x['fit']['NaCl 0.9']['slope'], 2), fmt(kk.pct_200.min()), fmt(kk.pct_200.max())))
    P('')
    # ---------------- f
    P('## f. KCl vs NaCl at 0.7-0.9 mol/kg')
    P('')
    P('Method: for each KCl isotherm, the NaCl 0.9 mol/kg linear fit of (a) evaluated at the KCl temperature, and the '
      'nearest-temperature NaCl isotherm. KCl 0.70 mol/kg is 0.69977 mol/kg; NaCl 0.9 is 0.902.')
    P('')
    P(md_table(pd.DataFrame(x['kcl_vs_nacl_pairs'], columns=['Tk', 'pk', 'Tn', 'pn']).assign(
        fit=[v[2] for v in x['kcl_vs_nacl']], d=[v[3] for v in x['kcl_vs_nacl']]),
        ['Tk', 'pk', 'Tn', 'pn', 'fit', 'd'],
        ['KCl T (C)', 'KCl % to 200', 'nearest NaCl T (C)', 'NaCl % to 200', 'NaCl fit at KCl T (%)', 'KCl - NaCl fit (pp)'],
        [2, 2, 2, 2, 2, 2]))
    dd_ = np.array([v[3] for v in x['kcl_vs_nacl']])
    P('')
    P('KCl exceeds the NaCl fit by %s to %s pp (mean %s pp) at matched T (%s at every matched temperature); '
      'the nearest-isotherm pairs differ by up to %s K, which matters in the cold because of the %s pp/C slope.' % (
          fmt(dd_.min()), fmt(dd_.max()), fmt(dd_.mean()), 'KCl is larger' if (dd_ > 0).all() else 'KCl is NOT larger',
          fmt(max(abs(a - b) for a, _, b, _ in x['kcl_vs_nacl_pairs'])), fmt(abs(x['fit']['NaCl 0.9']['slope']), 2)))
    P('')
    P('Literature (kcl_literature_ratio.csv, interpolated to 200 MPa; all at 25 C and DILUTE, 0.0005-0.05 mol/L, no literature '
      'at 0.7 mol/kg): ' + '; '.join('%s %s M: +%s%%' % (a, c, fmt(v, 2)) for (a, c), (v, ph) in x['lit'].items()) + '.')
    k25 = t[(t.solution == 'KCl 0.70') & (abs(t.T_C - 24.9) < 0.5)].iloc[0]
    P('Our KCl 0.70 mol/kg at %s C: +%s%% (vs +%s%% for 0.02 M Fisher and Fox): %s pp difference, so the 0.7 mol/kg KCl ratio '
      'at 25 C is %s the dilute literature value within %s pp; no literature exists to compare the temperature dependence '
      '(all literature is at 25 C).' % (
          fmt(k25.T_C), fmt(k25.pct_200), fmt(x['lit'][('Fisher and Fox 1979', 0.02)][0], 2),
          fmt(k25.pct_200 - x['lit'][('Fisher and Fox 1979', 0.02)][0], 2),
          'close to' if abs(k25.pct_200 - x['lit'][('Fisher and Fox 1979', 0.02)][0]) < 0.5 else 'not equal to',
          fmt(abs(k25.pct_200 - x['lit'][('Fisher and Fox 1979', 0.02)][0]), 2)))
    P('')
    # ---------------- g
    P('## g. Data-quality caveats')
    P('')
    fac = res['factors']
    a25f = fac[fac.run.str.contains('Aug')]
    n17 = fac[fac.run.str.contains('Nov 17|Nov 18|Nov 19')]
    kf = fac[fac.run.str.contains('KCl')]
    o = x['offsets']
    k1, k2 = list(K_ALT)
    P('Plain summary. Absolute conductivities are anchored: each run is rescaled so its first 1-bar step equals prior 1-bar data '
      '(McCleskey 2011/2012), which keeps the measured pressure dependence but replaces the run\'s own calibration; the rescaling '
      'factors range from %s%% to %s%% (Aug 2025 0.5 mol/kg: %s%% to %s%%; Nov 17-19: %s%% to %s%%; KCl: %s%% to %s%%). '
      'After the Nov 2026 connection repair the cell constant changed (KCl standard %s -> 115.21 1/m, QUOTED) and the 1-bar NaCl '
      'readings sit below McCleskey (2011) by an amount that grows on cooling (QUOTED -2.7%% to -6.8%% with Diaz\'s K; recomputed '
      'below); its cause is not established. The 16.2 C anchor step may not have been thermally settled. The percent increase to '
      '200 MPa does not depend on the anchoring, but could depend on the offset\'s cause (bound below). Details:' % (
          fmt(fac['pct_change'].min(), 1, True), fmt(fac['pct_change'].max(), 1, True), fmt(a25f['pct_change'].min(), 1, True),
          fmt(a25f['pct_change'].max(), 1, True), fmt(n17['pct_change'].min(), 1, True), fmt(n17['pct_change'].max(), 1, True),
          fmt(kf['pct_change'].min(), 1, True), fmt(kf['pct_change'].max(), 1, True), fmt(x['K_kcl_pre'], 2)))
    P('')
    P('**Anchoring.** Every run is multiplied by one factor so its first ascending 1-bar step equals McCleskey (2011), or '
      'McCleskey et al. (2012) where T < 5 C. The factors (reference/measured) are: Aug 2025 0.5 mol/kg %s to %s (%s%% to %s%%); '
      'Oct 7-8 and Nov 16 exactly 1 by construction of their cell constants (K = %s and %s 1/m; Oct 7-8 uses the Oct 14 KCl '
      'estimate); Nov 17-19 %s to %s (%s%% to %s%%) with ONE K = %s 1/m (Nov 17 24.8 C = 1 by construction); KCl 0.70 mol/kg '
      '%s to %s (%s%% to %s%%, pre-repair K = %s 1/m). The measured pressure dependence is kept; the ratio is unchanged. The '
      'anchor uncertainty used in error bars (1%% MC2011, 2%% MC12) is an assumption, not derived. '
      '**Post-repair 1-bar offset (QUOTED from open-questions 2026-11-20 and recomputed here with Diaz\'s K = 116.84):** '
      'with that K the Nov 17-19 NaCl readings fall below McCleskey (2011) by %s%% at %s C growing to %s%% at %s C '
      '(recomputed: %s). With the Nov 19 KCl standard K = 115.21 (QUOTED) the offsets would be %s to %s%%. **Cell constant '
      'across the repair:** KCl standard K = %s 1/m before (Nov 5-12, computed from the CSV) vs 115.21 after (QUOTED) a change '
      'of %s%%; the Nov 17-19 NaCl series uses K = %s (self-calibrated on the 24.8 C 1-bar step vs McCleskey), %s%% away '
      'from the post-repair standard value, so the absolute scale of the Nov 17-19 data rests on that choice and on the '
      'anchoring. If the offset were a series resistance rather than a scale factor it would dilute the measured pressure '
      'effect: under that hypothesis (K = %s taken as right) the 200 MPa increases would be %s instead of %s (percent, 24.8 to 2 C '
      'respectively); this is a bound, not a result. **16.2 C anchor:** the notebook (2026-11-23 entry) says the first 1-bar '
      'step of that isotherm was taken before temperature settled; anchored instead on its closing 1-bar step the 200 MPa '
      'increase would be %s%% rather than %s%% (the closing step is %s%% relative to the first step). '
      '**Exclusions:** Aug 2025 steps are excluded for bench checks, drifts and hasty decompressions (EXCLUDED in '
      'plot_nacl_sigma_P.py; Aug 14 entirely); the 2026 Nov 12 +/-12 ohm CPE-fit steps and Nov 19 out-of-vessel sweeps '
      'are excluded; Oct 6, Oct 9, Oct 14-15 are not in the figure. Nothing is removed from the configs.' % (
          fmt(a25f.factor.min(), 3), fmt(a25f.factor.max(), 3), fmt(a25f['pct_change'].min(), 1, True),
          fmt(a25f['pct_change'].max(), 1, True), fmt(x['K_oct'], 2), fmt(x['K_nov16'], 2), fmt(n17.factor.min(), 3),
          fmt(n17.factor.max(), 3), fmt(n17['pct_change'].min(), 1, True), fmt(n17['pct_change'].max(), 1, True),
          fmt(x['K_nov17'], 3), fmt(kf.factor.min(), 3), fmt(kf.factor.max(), 3), fmt(kf['pct_change'].min(), 1, True),
          fmt(kf['pct_change'].max(), 1, True), fmt(x['K_kcl_pre'], 2),
          fmt(o['off_' + k1].iloc[0], 1), fmt(o['T'].iloc[0]), fmt(o['off_' + k1].min(), 1), fmt(o['T'].iloc[o['off_' + k1].argmin()]),
          ', '.join('%s C: %s%%' % (fmt(T), fmt(v, 1)) for T, v in zip(o['T'], o['off_' + k1])),
          fmt(o['off_' + k2].max(), 1, True), fmt(o['off_' + k2].min(), 1, True),
          fmt(x['K_kcl_pre'], 2), fmt(100 * (115.21 / x['K_kcl_pre'] - 1), 1, True), fmt(x['K_nov17'], 3),
          fmt(100 * (x['K_nov17'] / 115.21 - 1), 1, True), fmt(x['K_nov17'], 3),
          ', '.join(fmt(s[3]) for s in x['series']), ', '.join(fmt(s[2]) for s in x['series']),
          fmt(x['alt_anchor'][[i for i, a in enumerate(x['alt_anchor']) if '16.2' in a[0]][0]][2]),
          fmt(x['alt_anchor'][[i for i, a in enumerate(x['alt_anchor']) if '16.2' in a[0]][0]][1]),
          fmt(100 * (res['d'][(res['d'].run.str.contains('16.2')) & res['d'].descent & (res['d'].P_MPa < 3)]
                     .sigma_Sm.iloc[0] / t[t.run.str.contains('16.2')].sigma_1bar_anch.iloc[0] - 1), 2, True)))
    P('')
    P('Anchoring factor table (main figure):')
    P('')
    P(md_table(fac, ['run', 'T_C', 'sigma_meas_1bar', 'sigma_ref', 'ref_source', 'factor', 'pct_change'],
               ['run', 'T (C)', 'measured 1 bar (S/m)', 'reference (S/m)', 'source', 'factor', '% change'],
               [None, 2, 4, 4, None, 4, 2]))
    P('')
    P('Series-resistance hypothesis, Nov 17-19 (illustrative bound): ' + '; '.join(
        '%s: %s%% -> %s%%' % (r_[0].replace('NaCl 0.9 m, ', ''), fmt(r_[2]), fmt(r_[3])) for r_ in x['series']) + '.')
    P('')
    P('Alternative anchor on the closing 1-bar step (pct to 200 MPa; closing step P in MPa): ' + '; '.join(
        '%s: %s%% -> %s%% (%s MPa)' % (a[0].replace('NaCl 0.9 m, ', ''), fmt(a[1]), fmt(a[2]), fmt(a[3], 2)) for a in x['alt_anchor']) + '.')
    P('')
    # ---------------- h
    P('## h. Not established / inconsistent')
    P('')
    t10 = t[abs(t.T_C - 9.85) < 0.2].iloc[0]
    t0 = t[abs(t.T_C - 0.85) < 0.2].iloc[0]
    t35 = t[abs(t.T_C - 34.85) < 0.2].iloc[0]
    L2 = [
        'Aug 2025 0.5 mol/kg increase to 200 MPa (%s-%s %%, excluding 9.85 C) is about %sx the 0.9 mol/kg increase at comparable T '
        '(%s%% at 20.6 C, %s%% at 24.8 C) and about %sx Adams and Hall at 25 C (+%s%%). Not resolved: concentration effect, '
        'the first in-vessel 1-bar steps of Aug 2025, or the older vessel/anchoring. This is the largest inconsistency.' % (
            fmt(t[(t.solution == 'NaCl 0.5') & (abs(t.T_C - 9.85) > .5)].pct_200.min()),
            fmt(t[(t.solution == 'NaCl 0.5') & (abs(t.T_C - 9.85) > .5)].pct_200.max()),
            fmt(t[(t.solution == 'NaCl 0.5') & (abs(t.T_C - 20) < 2)].pct_200.mean() /
                t[t.run.str.contains('20.5 C')].pct_200.iloc[0], 1),
            fmt(t[t.run.str.contains('20.5 C')].pct_200.iloc[0]), fmt(t[t.run.str.contains('24.8 C')].pct_200.iloc[0]),
            fmt(t[(t.solution == 'NaCl 0.5') & (abs(t.T_C - 20) < 2)].pct_200.mean() / ah['pct_200'], 1), fmt(ah['pct_200'])),
        'NaCl 0.5 9.85 C: +%s%% at 200 MPa (+%s%% step range) is far off the trend of its neighbours (0.85 C: +%s%%, 20 C: +%s%%); '
        'its 1-bar step has a wide sweep range (anchor factor %s; the other 0.5 mol/kg isotherms at <= 20.2 C have %s%% to %s%%) '
        'and the first 50 MPa step rises by %s%% against %s%% at 0.85 C and %s%% at 34.85 C. Looks like an unsettled '
        '1-bar step; not changed here.' % (
            fmt(t10.pct_200), fmt(t10.pct_200_steprange), fmt(t0.pct_200),
            fmt(t[t.run.str.contains('Aug 13')].pct_200.iloc[0]), fmt(t10.anchor_factor, 3),
            fmt(fac[fac.run.str.contains('Aug') & (fac.T_C < 20.3) & (abs(fac.T_C - 9.85) > .5)]['pct_change'].min(), 1, True),
            fmt(fac[fac.run.str.contains('Aug') & (fac.T_C < 20.3) & (abs(fac.T_C - 9.85) > .5)]['pct_change'].max(), 1, True),
            fmt(100 * (res['d'][(res['d'].run == t10.run) & (res['d'].P_MPa.between(40, 60))].sigma_Sm.iloc[0] /
                       t10.sigma_1bar_anch - 1)),
            fmt(100 * (res['d'][(res['d'].run == t0.run) & (res['d'].P_MPa.between(40, 60))].sigma_Sm.iloc[0] /
                       t0.sigma_1bar_anch - 1)),
            fmt(100 * (res['d'][(res['d'].run == t35.run) & (res['d'].P_MPa.between(40, 60))].sigma_Sm.iloc[0] /
                       t35.sigma_1bar_anch - 1))),
        'The three nominal-20 C 0.5 mol/kg isotherms (Aug 13: %s C, Aug 18: %s C, Aug 20: %s C) differ in T, and at ~600 MPa their '
        'ratio to 1 bar is %s; Aug 20 20 C has no 200 MPa value.' % (
            fmt(t[t.run.str.contains('Aug 13')].T_C.iloc[0]), fmt(t[t.run.str.contains('Aug 18')].T_C.iloc[0]),
            fmt(t[t.run.str.contains('Aug 20, 20')].T_C.iloc[0]),
            ', '.join('%s%%' % fmt(pc, 1, True) for run, T, Pq, pc, f_, pm_ in x['at600'] if abs(T - 20) < 2)),
        'No kept decompression step exists for 0.5 mol/kg, so no up/down repeatability can be stated for the Aug 2025 data; '
        'its error bar in the figures is the step range only.',
        'Anchor uncertainties (1%% / 2%%) are assumed, not derived; a 1%% error in the 1-bar anchor changes the %s%% increase '
        'by about 0.1 pp but changes absolute sigma by 1%%. Pressure gauge calibration and temperature accuracy (bath vs '
        'sample) are not quantified here.' % fmt(n9.pct_200.iloc[-1], 1),
        'Post-repair 1-bar offset on cooling (g) has no established cause; the Nov 17-19 absolute scale and, if the cause is a '
        'series resistance, the size of the cold increases (by up to the bound in g) are not settled. K = %s (self-calibrated) vs '
        '115.21 (Nov 19 standard, QUOTED) are %s%% apart.' % (fmt(x['K_nov17'], 3), fmt(100 * (x['K_nov17'] / 115.21 - 1), 1)),
        'Pan et al. (2021) vs our 0.5 mol/kg data inside their pressure range (>= 212 MPa, T <= 295 K): our absolute sigma is above '
        'the model by %s%% to %s%% (see d). Not understood; note the model\'s 1-bar values are also far from the reference '
        '(%s%% to %s%% for 0.5 mol/kg). Not investigated further here.' % (
            fmt(min(v[4] for v in x['pan_inside']), 0, True), fmt(max(v[5] for v in x['pan_inside']), 0, True),
            fmt(x['pan'][x['pan'].solution == 'NaCl 0.5'].pan_1bar_vs_ref.min(), 1, True),
            fmt(x['pan'][x['pan'].solution == 'NaCl 0.5'].pan_1bar_vs_ref.max(), 1, True)),
        'Pan et al. 1-bar values are %s%% to %s%% from the 1-bar reference (outside their data), so only their pressure RATIO '
        'is compared here.' % (fmt(x['pan'].pan_1bar_vs_ref.min(), 1, True), fmt(x['pan'].pan_1bar_vs_ref.max(), 1, True)),
        'Adams and Hall transcription awaits an independent check (CSV header); their absolute sigma uses McCleskey (2012) at 25 C. '
        'Their percent increase does not depend on that.',
        'The abstract\'s "274-318 K" does not match the kept data (%s-%s K).' % (fmt(T25[2] + 273.15), fmt(T25[3] + 273.15)),
        'Not checked: colour-vision distinguishability by simulation beyond the palette choice described in README.md; '
        'TEMP_COLORS of the main figure is not monotone cold-to-warm (violet 4-8 C, pink 24-27 C), so the poster figures use a '
        'new monotone palette (not the TEMP_COLORS bins).',
    ]
    for item in L2:
        P('- ' + item)
    P('')
    open(os.path.join(HERE, 'poster_facts.md'), 'w').write('\n'.join(L) + '\n')
    cols = ['run', 'solution', 'source', 'date_first', 'date_last', 'T_C', 'T_1bar_step_C', 'P_1bar_step_MPa', 'n_steps',
            'n_down', 'anchor_source', 'anchor_factor', 'sigma_1bar_anch', 'sigma_200_anch', 'method_200', 'pct_200',
            'pct_200_unanchored', 'pct_200_steprange', 'ud_max_pp', 'ud_mean_pp', 'ud_rms_pp', 'return_1bar_pct',
            'P_max_step_MPa', 'P_max_parabola_MPa', 'P_max_guide_MPa', 'pct_at_max_step', 'P_hi_MPa', 'pct_at_Phi',
            'pan_pct_200', 'pan_envelope']
    t[cols].round(5).to_csv(os.path.join(HERE, 'poster_facts.csv'), index=False)
    print('\n'.join(L))


if __name__ == '__main__':
    write(compute())
