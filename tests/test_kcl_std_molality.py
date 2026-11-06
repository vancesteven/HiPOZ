"""KCl standard bottle ppt values are g per kg water (Steve, 2026-11-06)."""
import pytest

from gamryTools import kcl_std_molal, wKCl_ppt, get_w_from_descrip
from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012
import numpy as np


def test_80mS_bottle_molality_is_per_kg_water():
    assert kcl_std_molal(52.168) == pytest.approx(0.69977, rel=1e-4)


def test_80mS_bottle_matches_its_label_under_mccleskey():
    m = kcl_std_molal(wKCl_ppt[80000])
    ions = {'K_p1': {'mols': np.array([m])}, 'Cl_m1': {'mols': np.array([m])}}
    sigma = elecCondMcCleskey2012(25.0, ions)['sigma_Sm'][0][0]
    assert sigma == pytest.approx(8.0, rel=0.005)


def test_descrip_label_uses_water_basis():
    comp, w_ppt, w_molal = get_w_from_descrip('KCl standard', lbl_uScm=80000)
    assert (comp, w_ppt) == ('KCl', 52.168)
    assert w_molal == pytest.approx(0.69977, rel=1e-4)
