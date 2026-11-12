#!/usr/bin/env python
"""
Concentration bookkeeping for the Cortes et al. (2026) benchtop data.

  * benchtop concentrations are mol/kg as prepared: the lab makes only molal
    solutions, so the raw headers "0.5 M/L" are molalities (Steve,
    2026-11-07; this reverses the 2026-10-17 molarity conversion);
  * mixture ion specs (2026-10-17) are per mole of the FIRST component, because the
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


def test_committed_csv_is_labeled_molality():
    """The committed benchtop CSV carries the prepared molality, unconverted."""
    df = pd.read_csv(REPO / 'cortes2026' / 'Cortes2026BenchtopData.csv', comment='#')
    assert 'concentration_molar' not in df.columns
    labeled = df.notes.str.extract(r'^([\d.]+)', expand=False).astype(float)
    nacl = df[(df.compound == 'NaCl') & labeled.notna()]
    assert len(nacl) > 0
    assert (nacl.concentration_molal == labeled[nacl.index]).all()
