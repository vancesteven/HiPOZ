#!/usr/bin/env python
"""
Anchor tests for the McCleskey (2012) conductivity implementation.

The lambda_i(T, I) term must be evaluated at the TOTAL effective ionic
strength of the solution (I = 0.5 * sum_i m_i z_i^2), not at a per-ion
partial value. With total I, the model reproduces KCl conductivity
standards within its stated ~1% accuracy; with per-ion I it drifts to
+4.6% at 1 molal (caught 2026-10-05).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012
import cortes_mccleskey as cm


# (molality, sigma S/m) KCl conductivity standards at 25 C
KCL_STANDARDS = [(0.01, 0.1413), (0.1, 1.2890), (1.0, 11.19)]


@pytest.mark.parametrize('m,sigma_std', KCL_STANDARDS)
def test_kcl_standards_within_1pct(m, sigma_std):
    ions = {'K_p1': {'mols': np.array([m])}, 'Cl_m1': {'mols': np.array([m])}}
    sigma = elecCondMcCleskey2012(25.0, ions)['sigma_Sm'][0][0]
    assert sigma == pytest.approx(sigma_std, rel=0.01), \
        f'KCl {m} molal: {sigma:.4f} S/m vs standard {sigma_std}'


def test_lambda_uses_total_ionic_strength():
    """Each ion's lambda must see I from ALL ions, not only its own."""
    m = 1.0
    both = {'K_p1': {'mols': np.array([m])}, 'Cl_m1': {'mols': np.array([m])}}
    elecCondMcCleskey2012(25.0, both)
    # For 1 molal KCl the total I is 1.0; per-ion I would be 0.5
    np.testing.assert_allclose(both['K_p1']['I'], [1.0])
    np.testing.assert_allclose(both['Cl_m1']['I'], [1.0])


def test_mgso4_total_I_counts_charge_squared():
    m = 0.5
    ions = {'Mg_p2': {'mols': np.array([m])}, 'SO4_m2': {'mols': np.array([m])}}
    elecCondMcCleskey2012(25.0, ions)
    # I = 0.5 * (m*4 + m*4) = 4m
    np.testing.assert_allclose(ions['Mg_p2']['I'], [4 * m])


def test_resolve_speciation_auto_without_reaktoro():
    """In an env without Reaktoro, 'auto' must fall back to False quietly."""
    import speciation as spec
    if spec.available():
        pytest.skip('Reaktoro installed; fallback path not exercised')
    assert cm.resolve_speciation('MgSO4', 'auto') is False
    assert cm.resolve_speciation('NaCl', 'auto') is False


def test_resolve_speciation_explicit_override():
    assert cm.resolve_speciation('MgSO4', False) is False
    assert cm.resolve_speciation('MgSO4', True) is True


def test_compute_for_data_auto_matches_off_without_reaktoro():
    import pandas as pd
    import speciation as spec
    if spec.available():
        pytest.skip('Reaktoro installed; auto is expected to differ from off')
    data = pd.DataFrame({'w_molal': [0.1, 1.0], 'T_K': [298.15, 298.15]})
    auto = cm.compute_mccleskey_for_data(data, 'KCl', speciation='auto')
    off = cm.compute_mccleskey_for_data(data, 'KCl', speciation=False)
    np.testing.assert_allclose(auto, off)
