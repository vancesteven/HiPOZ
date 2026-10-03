#!/usr/bin/env python
"""
Test the plot_sigma_PT.py CLI script against real repo data (headless).
"""

import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import pytest

import plot_sigma_PT


def test_map_from_curated_dates(tmp_path):
    rc = plot_sigma_PT.main([
        '--dates', '20260818', '20260827',
        '--sigma-ref', '8.0', '--no-tex',
        '--xtn', 'png',
        '--out-dir', str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / 'sigma_PT_map.png').exists()
    assert not (tmp_path / 'provisional_sigma.csv').exists(), \
        'No raw dates requested, so no provisional CSV expected'


def test_map_with_raw_dates(tmp_path):
    rc = plot_sigma_PT.main([
        '--dates', '20260827',
        '--raw-dates', '20260922',
        '--max-files', '2',
        '--sigma-ref', '8.0', '--no-tex',
        '--xtn', 'png',
        '--out-dir', str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / 'sigma_PT_map.png').exists()
    assert (tmp_path / 'raw_sigma_timeseries.png').exists()
    csv = pd.read_csv(tmp_path / 'provisional_sigma.csv')
    assert len(csv) <= 2
    assert {'date', 'P_MPa', 'T_K', 'sigma_Sm'}.issubset(csv.columns)
    # provisional sigma should be physically plausible for the KCl standard
    assert (csv.sigma_Sm > 0).all()


def test_curated_date_without_zanalysis_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_sigma_PT.load_curated(['20260922'])


def test_implied_k_cell_matches_curated_dataset():
    cur = plot_sigma_PT.load_curated(['20260827'])
    k = plot_sigma_PT.implied_k_cell(cur)
    # The Aug-Sep 2026 KCl study used a single shared cell constant
    assert k == pytest.approx(128.40, abs=0.05)
