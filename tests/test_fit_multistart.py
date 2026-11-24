"""Multi-start CPE fit in gamryTools.Solution.fit_circuit."""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import gamryTools  # noqa: E402
from gamryTools import Solution, select_best_candidate  # noqa: E402

F_RANGE = 1e3 * np.array([10, 100])
# Chained-guess refit of the known-bad sweep (cortes2026/diaz/refit_chained.csv)
BAD_NAME, BAD_CHAINED_R = 'Default_20261014_133735_P_124_T_296.txt', 14.5529
GOOD_NAME = 'Default_20261014_094111_P_2_T_296.txt'


def _find(name):
    hits = glob.glob(str(REPO / 'data' / name.split('_')[1] / 'Conductivity*' / name))
    if not hits:
        pytest.skip(f'data file {name} not present')
    return hits[0]


def _fit(name, multistart):
    s = Solution(cmap_name='viridis')
    s.load_file(_find(name))
    s.fit_circuit(circ_type='CPE', print_circuit=False, f_range_hz=F_RANGE, multistart=multistart)
    return s


def test_known_bad_sweep_reaches_good_minimum():
    old, new = _fit(BAD_NAME, False), _fit(BAD_NAME, True)
    assert abs(new.Rcalc_ohm / BAD_CHAINED_R - 1) < 2e-3
    assert abs(old.Rcalc_ohm / BAD_CHAINED_R - 1) > 2e-3   # old single start is in the bad minimum
    assert new.fit_start != 'default'
    assert new.Runc_ohm < old.Runc_ohm


def test_well_behaved_sweep_identical_to_single_start():
    old, new = _fit(GOOD_NAME, False), _fit(GOOD_NAME, True)
    assert new.fit_start == 'default'
    assert new.Rcalc_ohm == old.Rcalc_ohm
    assert new.Runc_ohm == old.Runc_ohm


def test_selection_picks_lower_residual_candidate():
    stub = object()
    cands = [('default', stub, 1.0), ('a', 'A', 0.5), ('b', 'B', 0.2), ('c', 'C', float('nan'))]
    assert select_best_candidate(cands)[0] == 'b'
    # a candidate that is only marginally better does not displace the default
    assert select_best_candidate([('default', stub, 1.0), ('a', 'A', 0.995)])[0] == 'default'
    # a worse candidate never wins
    assert select_best_candidate([('default', stub, 1.0), ('a', 'A', 2.0)])[0] == 'default'


def test_multistart_is_opt_in(monkeypatch):
    """Default calls keep the single-start result; the module flag (set by
    `--multistart`) switches the extra starts on."""
    import gamryTools
    name = BAD_NAME
    s = Solution(cmap_name='viridis')
    s.load_file(_find(name))
    s.fit_circuit(circ_type='CPE', print_circuit=False, f_range_hz=F_RANGE)
    assert s.fit_start == 'default'
    assert s.Rcalc_ohm == _fit(name, False).Rcalc_ohm

    monkeypatch.setattr(gamryTools, 'MULTISTART_DEFAULT', True)
    s2 = Solution(cmap_name='viridis')
    s2.load_file(_find(name))
    s2.fit_circuit(circ_type='CPE', print_circuit=False, f_range_hz=F_RANGE)
    assert s2.fit_start != 'default'
    assert s2.Rcalc_ohm == _fit(name, True).Rcalc_ohm
