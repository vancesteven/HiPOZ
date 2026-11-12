"""Pan et al. (2021) NaCl conductivity reproduces the authors' regression."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pan2021_nacl as pan


def _rows():
    with open(REPO / 'tests' / 'data' / 'pan2021_validation.csv') as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def test_reproduces_authors_sigma_calc():
    rows = _rows()
    assert len(rows) >= 60
    resid = np.abs([np.log10(pan.sigma_Sm(r['molality_molkg'], r['T_K'], r['rho_gcm3']))
                    - np.log10(r['sigma_calc_Sm']) for r in rows])
    assert np.median(resid) < 1e-10
    assert resid.max() < 0.03


def test_seafreeze_density_path_matches_dataset():
    for r in _rows()[:10]:
        s = pan.sigma_vs_P(r['molality_molkg'], r['T_K'], r['P_MPa'])[0]
        assert np.log10(s) == pytest.approx(np.log10(r['sigma_calc_Sm']), abs=0.01)


def test_envelope():
    assert pan.in_envelope(300.0, 290.0)
    assert not pan.in_envelope(100.0, 290.0)
    assert not pan.in_envelope(300.0, 308.0)
