"""vessel_1bar_check: prior-data NaCl interpolation, equilibration flag, Nov 16 K."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'cortes2026'))
sys.path.insert(0, REPO)
import vessel_1bar_check as vbc  # noqa: E402


def test_prior_reproduces_table_nodes():
    # At a tabulated molality and temperature the interpolation returns the table
    assert vbc.sigma_prior_nacl(0.5, 25.0) == pytest.approx(4.652, rel=1e-9)
    assert vbc.sigma_prior_nacl(0.9999, 35.0) == pytest.approx(10.19, rel=1e-9)
    with pytest.raises(ValueError):
        vbc.sigma_prior_nacl(1.5, 25.0)


def test_prior_close_to_mc12_at_nov16():
    # 0.902 m, 29.3 C: McCleskey (2011) data and the MC12 model agree to < 1 %
    s = vbc.sigma_prior_nacl(0.902, 29.3)
    mc = vbc._mc12(29.3, {'Na_p1': 0.902, 'Cl_m1': 0.902})
    assert abs(s / mc - 1) < 0.01


def test_flags_and_nov16_calibration():
    st = vbc.ambient_steps(vbc.load_sweeps())
    flag = dict(zip(st.start.dt.strftime('%m%d_%H%M'), st.recent_T_change))
    assert flag['1111_1527'] == 'possible'   # 2.5 h bracket after 6 -> 29 C
    assert flag['1110_1207'] == 'possible'
    assert flag['1112_1157'] == 'no'
    n, K = vbc.nacl_calibration(st)
    assert K == pytest.approx(120.4, abs=0.2)  # not the config's 92.7
    assert abs(n.dK_pct.iloc[-1]) < 0.5         # closing 1-bar step
