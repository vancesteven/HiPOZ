"""sigma_P_sources: absolute literature, 1-bar reference hierarchy, anchoring."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'cortes2026'))
sys.path.insert(0, REPO)
import sigma_P_sources as S  # noqa: E402


def test_fisher_fox_002M_absolute():
    d = S.fisher_fox_kcl(0.02)
    assert d.sigma_Sm.iloc[0] == pytest.approx(0.2767, abs=0.0005)
    s2000 = np.interp(2000 * S.ATM_MPA, d.P_MPa, d.sigma_Sm)
    assert s2000 == pytest.approx(0.2767 * 1.0787, rel=0.005)


def test_fisher_1962_002M_consistent_with_fisher_fox():
    f62, ff = S.fisher_1962_kcl(0.02), S.fisher_fox_kcl(0.02)
    P = 2000 * S.ATM_MPA
    assert np.interp(P, f62.P_MPa, f62.sigma_Sm) == pytest.approx(
        np.interp(P, ff.P_MPa, ff.sigma_Sm), rel=0.005)


def test_sigma_ref_1bar_hierarchy():
    v, src = S.sigma_ref_1bar('NaCl', 0.902, 29.3)
    assert v == pytest.approx(8.426, abs=0.01) and src == 'McCleskey (2011)'
    assert S.sigma_ref_1bar('KCl', 0.70, 3.0)[1] == 'McCleskey et al. (2012)'
    assert S.sigma_ref_1bar('KCl', 0.5, 25.0)[0] == pytest.approx(5.920, rel=1e-6)


def test_anchor_factor_nov16_is_unity():
    g = S.our_steps_2026(False)
    g = g[g.run == 'NaCl 0.9 m, Nov 16']
    f, scaled = S.anchor_to_1bar(g)
    assert f == pytest.approx(1.0, abs=0.002)
    r = S.with_ratio(g).ratio.to_numpy()
    assert (S.with_ratio(scaled).ratio.to_numpy() == pytest.approx(r))  # shape kept


def test_nov19_out_of_vessel_sweeps_excluded():
    """Nov 19 >= 16:44 is the sample holder in an external chiller (Diaz
    notebook p. 35): every sweep is excluded, and each group carries the
    notebook's identification (5 wt% NaCl, KCl true standard at 17:21,
    10 wt% NaCl at 17:50). None reaches the kept steps."""
    sw = pd.read_csv(S.SWEEPS_2026_CSV, parse_dates=['time'], keep_default_na=False)
    late = sw[sw.time >= '2026-11-19 16:44']
    assert len(late) == 12 and late.excluded.str.startswith('out-of-vessel check').all()
    hh = late.filename.str.slice(17, 23)
    assert late[hh.str.startswith('1644') | hh.str.startswith('1645') | hh.str.startswith('1646')].excluded.str.contains('5 wt% NaCl').all()
    assert late[hh.str.startswith('172')].excluded.str.contains('True Standard at 5:21 pm').all()
    assert (late[hh.str.startswith('175')].excluded.str.contains('10 wt% NaCl').sum() == 3)
    steps = pd.read_csv(S.STEPS_2026_CSV, parse_dates=['time'])
    assert not (steps.time >= '2026-11-19 16:44').any()

def test_nov17_first_1bar_anchor_is_unity_in_supplement_K():
    g = S.our_steps_2026(False)
    g = g[g.run == 'NaCl 0.9 m, Nov 17, 24.8 C']
    f, _ = S.anchor_to_1bar(g)
    assert f == pytest.approx(1.000, abs=0.002)
    # one K for all six isotherms
    k = pd.read_csv(S.STEPS_2026_CSV)
    assert k[k.run.str.contains('Nov 1[789]')].K_cell.nunique() == 1
