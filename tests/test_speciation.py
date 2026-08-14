#!/usr/bin/env python3
"""
Tests for WATEQ4F speciation of the McCleskey (2012) conductivity model.

Reference free-ion fractions below were produced by an independent
phreeqpython + IPhreeqc 3.4.0 run against the same wateq4f.dat database and
agreed with Reaktoro to every printed digit -- they are cross-validated, not
self-generated.

Tests needing Reaktoro skip cleanly when it is unavailable (it is conda-forge
only). The speciation=False path is tested unconditionally, since it must keep
working with no Reaktoro installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import speciation as spec
from study_plots import compute_mccleskey_model

requires_reaktoro = pytest.mark.skipif(
    not spec.available(), reason='Reaktoro not installed (conda-forge only)')

# Cross-validated against phreeqpython + IPhreeqc 3.4.0.
MGSO4_FREE_FRACTION_25C = {0.0249: 0.590, 1.6616: 0.257}


def test_default_path_needs_no_reaktoro():
    """speciation=False must work in an interpreter without Reaktoro."""
    out = compute_mccleskey_model(np.array([0.1, 1.0]), np.array([298.15]),
                                  compound='NaCl')
    assert len(out) == 1
    assert len(np.atleast_1d(out[0])) == 2
    assert np.all(np.atleast_1d(out[0]) > 0)


def test_return_shape_unchanged_by_kwarg():
    """The list[ndarray] contract that 13 call sites rely on must hold."""
    concs, temps = np.array([0.1, 0.5, 1.0]), np.array([273.15, 298.15])
    out = compute_mccleskey_model(concs, temps, compound='NaCl')
    assert isinstance(out, list) and len(out) == len(temps)
    for row in out:
        assert len(np.atleast_1d(row)) == len(concs)


def test_speciation_requires_known_compound():
    """A bare ion_spec cannot be speciated -- recipes are per-compound."""
    with pytest.raises(ValueError):
        compute_mccleskey_model(np.array([0.1]), np.array([298.15]),
                                ion_spec={'Na_p1': 1.0, 'Cl_m1': 1.0},
                                speciation=True)


@requires_reaktoro
@pytest.mark.parametrize('m,expected', sorted(MGSO4_FREE_FRACTION_25C.items()))
def test_mgso4_free_fraction(m, expected):
    """MgSO4 association must match the cross-validated PHREEQC result."""
    got = spec.free_fraction('MgSO4', m, 298.15)
    assert abs(got - expected) < 0.002, f'{got:.4f} vs {expected:.4f}'


@requires_reaktoro
def test_mgso4_association_increases_with_concentration():
    """Free fraction must fall monotonically as total molality rises."""
    fracs = [spec.free_fraction('MgSO4', m, 298.15)
             for m in (0.025, 0.33, 0.66, 1.0, 1.66)]
    assert all(b < a for a, b in zip(fracs, fracs[1:])), fracs


@requires_reaktoro
def test_neutral_species_excluded():
    """Neutral MgSO4(aq) must never appear in the conductivity sum."""
    ions = spec.free_ion_molalities('MgSO4', 1.0, 298.15)
    assert ions, 'speciation returned nothing'
    for key in ions:
        assert key.endswith(('_p1', '_p2', '_p3', '_m1', '_m2', '_m3')), key
    # MgSO4(aq) has no McCleskey parameters at all -- confirm the table agrees.
    from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012
    probe = {'MgSO4': {'mols': np.array([1.0])}}
    out = elecCondMcCleskey2012(25.0, probe)['sigma_Sm']
    assert float(np.atleast_1d(out).ravel()[0]) == 0.0


@requires_reaktoro
@pytest.mark.parametrize('compound', ['NaCl', 'KCl'])
def test_fully_dissociated_salts_are_a_no_op(compound):
    """
    WATEQ4F returns 1:1 chloride salts fully dissociated, so speciation must
    leave them alone. This protects the conductivity-standard calibration path.

    Agreement is close but not exact: speciation also accounts for water's own
    H+/OH- (~1.1e-7 molal), which the total-molality path omits. That shifts
    sigma by well under 0.1%.
    """
    concs, temps = np.array([0.01, 0.1, 1.0]), np.array([298.15])
    plain = compute_mccleskey_model(concs, temps, compound=compound)[0]
    spec_ = compute_mccleskey_model(concs, temps, compound=compound,
                                    speciation=True)[0]
    for a, b in zip(np.atleast_1d(plain), np.atleast_1d(spec_)):
        assert abs(b - a) / a < 5e-3, f'{compound}: {a:.6f} vs {b:.6f}'


@requires_reaktoro
def test_speciation_lowers_mgso4_conductivity():
    """
    Association must reduce predicted MgSO4 conductivity, and substantially so
    at high concentration where most of the salt is the neutral pair.
    """
    concs, temps = np.array([1.6616]), np.array([298.15])
    plain = float(np.atleast_1d(
        compute_mccleskey_model(concs, temps, compound='MgSO4')[0])[0])
    spec_ = float(np.atleast_1d(
        compute_mccleskey_model(concs, temps, compound='MgSO4',
                                speciation=True)[0])[0])
    assert spec_ < plain
    assert spec_ / plain < 0.5, f'expected a large drop, got {spec_ / plain:.3f}'


@requires_reaktoro
def test_speciation_is_idempotent():
    """Repeated calls must agree: the cache must not be mutated in place."""
    args = (np.array([0.3, 1.0]), np.array([298.15]))
    first = compute_mccleskey_model(*args, compound='MgSO4', speciation=True)[0]
    second = compute_mccleskey_model(*args, compound='MgSO4', speciation=True)[0]
    assert np.allclose(np.atleast_1d(first), np.atleast_1d(second))


@requires_reaktoro
def test_zero_concentration_returns_empty():
    assert spec.free_ion_molalities('MgSO4', 0.0, 298.15) == {}
