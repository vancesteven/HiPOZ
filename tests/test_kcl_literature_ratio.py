"""KCl kappa(P)/kappa(1 atm) literature loaders in plot_nacl_sigma_P."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'cortes2026'))
sys.path.insert(0, REPO)
import plot_nacl_sigma_P as pp  # noqa: E402


def test_fisher_fox_002M_2000atm():
    d = pp.load_fisher_fox_kcl(200.0)
    assert d.ratio.iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert np.interp(2000 * pp.ATM_MPA, d.P_MPa, d.ratio) == pytest.approx(1.0787, abs=0.002)


def test_adams_hall_kcl_258bar():
    a = pp.load_adams_hall_kcl()
    row = a[a.P_bar == 258].iloc[0]
    assert row.ratio == pytest.approx(1 / 0.98027)
    assert row.P_MPa == pytest.approx(25.8)
    assert a.descent.tolist()[-3:] == [True] * 3 and not a.descent.iloc[9]


def test_fisher_1962_001M_2000atm():
    a, _ = pp.load_fisher_1962_kcl()
    assert np.interp(2000 * pp.ATM_MPA, a.P_MPa, a.ratio) == pytest.approx(1.072)
