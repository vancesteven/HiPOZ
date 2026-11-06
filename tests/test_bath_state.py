"""Chiller-bath rows (P_MPa = 0 in the config) take P and T from the config,
not from the vessel readings in the Gamry file header."""
from types import SimpleNamespace

import numpy as np

from analysis_config import AnalysisConfig, bath_state
from headless_analysis import _state


def test_bath_row_uses_config_state():
    assert bath_state({'P_MPa': 0, 'T_K': 272.15}) == (0.0, 272.15)
    assert bath_state({'P_MPa': '0.0', 'T_K': '278.15'}, header_P_MPa=2.3) == (0.0, 278.15)


def test_vessel_rows_are_not_bath_rows():
    assert bath_state({'P_MPa': 100, 'T_K': 296.0}) is None
    # Config says 0 but the vessel was at 137 MPa (data/20260818 standard)
    assert bath_state({'P_MPa': 0.0, 'T_K': 295.76}, header_P_MPa=137.0) is None
    # Unknown temperature (legacy 'P_0_T_0' configs)
    assert bath_state({'P_MPa': 0, 'T_K': 0}) is None
    assert bath_state({'P_MPa': 0}) is None
    assert bath_state('file.txt') is None


def test_headless_state_prefers_bath_then_header():
    sol = SimpleNamespace(P_MPa=2.34, T_K=301.45)
    assert _state(sol, {'P_MPa': 0, 'T_K': 298.15}) == (0.0, 298.15)
    assert _state(sol, {'P_MPa': 100, 'T_K': 296}) == (2.34, 301.45)
    sol_hp = SimpleNamespace(P_MPa=137.0, T_K=295.8)
    assert _state(sol_hp, {'P_MPa': 0, 'T_K': 295.76}) == (137.0, 295.8)


def test_gui_metadata_carries_bath_state(tmp_path):
    cfg = tmp_path / 'zAnalysis20260923.csv'
    cfg.write_text(
        'group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes\n'
        'G,std.txt,0,298.15,standard,8.0,,,,,\n'
        'G,bath.txt,0,272.15,measurement,,NaCl,28.39,0.5,,\n'
        'G,vessel.txt,0,295.0,measurement,,NaCl,28.39,0.5,,\n')
    ts = SimpleNamespace(filenames=['std.txt', 'bath.txt', 'vessel.txt'],
                         Ps=np.array([2.0, 2.3, 150.0]),
                         conductivities_Sm=[None, None, None],
                         Rcalc_ohm=[14.99, 44.25, 30.0],
                         uncertainties=[0.3, 0.2, 0.2])
    ac = AnalysisConfig(str(cfg))
    ac.find_file_index = lambda f, t: t.filenames.index(f)
    result = ac._apply_calibration_group(ac.calibration_groups[0], ts)
    assert result['standard_metadata'][0]['T_K'] == 298.15
    bath, vessel = result['measurement_metadata']
    assert (bath['P_MPa'], bath['T_K']) == (0.0, 272.15)
    assert 'T_K' not in vessel and 'P_MPa' not in vessel
