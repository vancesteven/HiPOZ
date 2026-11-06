#!/usr/bin/env python
"""
Concentration bookkeeping for the Cortes et al. (2026) benchtop data.

Two corrections made 2026-10-17:
  * benchtop concentrations are mol/L (raw headers "0.5 M/L") and must be
    converted to mol/kg H2O before the molal McCleskey model is applied;
  * mixture ion specs are per mole of the FIRST component, because the
    benchtop concentration column carries the first component's
    concentration ('Na2SO4:KCl_2:1' at 1.5 = 1.5 M Na2SO4 + 0.75 M KCl).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'cortes2026'))

from cortes_mccleskey import get_mixture_ion_spec
import molarity_to_molality as m2m


@pytest.mark.parametrize('compound,expected', [
    ('Na2SO4:KCl_1:1', {'Na_p1': 2.0, 'SO4_m2': 1.0, 'K_p1': 1.0, 'Cl_m1': 1.0}),
    ('Na2SO4:KCl_2:1', {'Na_p1': 2.0, 'SO4_m2': 1.0, 'K_p1': 0.5, 'Cl_m1': 0.5}),
    ('Na2SO4:KCl_1:2', {'Na_p1': 2.0, 'SO4_m2': 1.0, 'K_p1': 2.0, 'Cl_m1': 2.0}),
    ('NaCl:MgSO4_1:1', {'Na_p1': 1.0, 'Cl_m1': 1.0, 'Mg_p2': 1.0, 'SO4_m2': 1.0}),
    ('NaCl:MgSO4_2:1', {'Na_p1': 1.0, 'Cl_m1': 1.0, 'Mg_p2': 0.5, 'SO4_m2': 0.5}),
    ('NaCl:MgSO4_1:2', {'Na_p1': 1.0, 'Cl_m1': 1.0, 'Mg_p2': 2.0, 'SO4_m2': 2.0}),
])
def test_mixture_ion_spec_is_per_first_component(compound, expected):
    assert get_mixture_ion_spec(compound) == pytest.approx(expected)


def test_non_mixture_has_no_spec():
    assert get_mixture_ion_spec('NaCl') is None
    assert get_mixture_ion_spec('NaCl+Glycine') is None


def test_mixture_components_from_label():
    assert m2m.mixture_components('Na2SO4:KCl_2:1', 1.5) == {'Na2SO4': 1.5, 'KCl': 0.75}
    assert m2m.mixture_components('NaCl:MgSO4_1:2', 0.75) == {'NaCl': 0.75, 'MgSO4': 1.5}
    assert m2m.mixture_components('NaCl', 1.0) == {'NaCl': 1.0}
    assert m2m.mixture_components('Glycine', 0.1) is None


def test_molality_always_exceeds_molarity():
    for (compound, molar), molal in m2m.MOLALITY_TABLE.items():
        assert molal > molar, f'{compound} {molar} M -> {molal} m'


def test_convert_frame_keeps_original_and_is_idempotent():
    df = pd.DataFrame({
        'compound': ['NaCl', 'KCl', 'Glycine', 'NaCl+Glycine'],
        'concentration_molal': [1.0, 1.5, 0.1, 0.5],
        'conductivity_Sm': [8.2, 15.2, 0.01, 4.4],
    })
    once = m2m.convert_frame(df)
    assert list(once['concentration_molar']) == [1.0, 1.5, 0.1, 0.5]
    assert list(once['concentration_molal']) == [1.0200, 1.5710, 0.1, 0.5054]
    twice = m2m.convert_frame(once)
    pd.testing.assert_frame_equal(once, twice)


def test_committed_csv_is_converted():
    """The committed benchtop CSV must carry true molality."""
    df = pd.read_csv(REPO / 'cortes2026' / 'Cortes2026BenchtopData.csv', comment='#')
    assert 'concentration_molar' in df.columns
    nacl = df[(df.compound == 'NaCl') & (df.concentration_molar == 1.0)]
    assert (nacl.concentration_molal == 1.0200).all()
