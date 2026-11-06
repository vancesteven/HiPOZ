"""Gamry header timestamps in both date styles the instrument writes."""
from datetime import datetime

import pytest

from gamryTools import Solution, parse_gamry_time


def test_iso_and_us_dates_parse_to_same_instant():
    iso = parse_gamry_time('2:49:18.272 PM 2026-11-05\n')
    us = parse_gamry_time('2:49:18.272 PM 11/5/2026\n')
    assert iso == us == datetime(2026, 11, 5, 14, 49, 18, 272000)


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        parse_gamry_time('Nov 5 2026 14:49')


def test_us_dated_file_loads():
    s = Solution(cmap_name='viridis')
    s.load_file('data/20261105/ConductivityData_Default/Default_20261105_113548_P_1_T_298.txt')
    assert s.time == datetime(2026, 11, 5, 11, 35, 48, 137000)
    assert s.P_MPa == pytest.approx(0.98)
