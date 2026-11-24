"""Checks for mahboub2026/mahboub_mc12_deviations.py."""
import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'mahboub2026'))
sys.path.insert(0, REPO)

import speciation  # noqa: E402

pytestmark = pytest.mark.skipif(not speciation.available(), reason='Reaktoro required')

import mahboub_mc12_deviations as md  # noqa: E402


@pytest.fixture(scope='module')
def result():
    return md.compute()[0]


def test_row_count_matches_modelled_data_rows(result):
    raw = pd.read_csv(md.DATA, comment='#')
    assert len(result) == raw.compound.isin(md.COMPOUNDS).sum() == 441
    assert result[['model_prefix_Sm', 'model_corrected_Sm']].notna().all().all()


def test_prefix_exceeds_corrected_nacl_1m():
    pre = md.prefix_model('NaCl', 1.0, 25.0)
    post = md.corrected_model('NaCl', 1.0, 25.0, True)
    assert pre > post
    assert 0.03 < (pre - post) / post < 0.08  # per-ion I overestimates by ~4-5%


def test_mgso4_speciation_lowers_sigma():
    # ion pairing removes charge carriers: speciated < fully dissociated
    full = md.corrected_model('MgSO4', 0.5, 25.0, False)
    assert md.corrected_model('MgSO4', 0.5, 25.0, True) < full


def test_in_range_flag(result):
    assert not result[result.T_C < 0].in_range.any()
    assert not result[result.I_eff > 0.7].in_range.any()
    assert result.in_range.any()
