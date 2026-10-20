"""Writing a config back (headless, GUI, --cal-map, harmonize) must not lose
rows, groups, exclusions, notes or extra columns."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

import config_store as cs
from analysis_config import AnalysisConfig

HEADER = ('group_name,filename,P_MPa,T_K,type,Z_Ohm,conductivity_Sm,comp,'
          'w_ppt,w_molal,exclude,notes\n')
ROWS = [
    'Morning,std.txt,0,298.15,standard,14.99,8.0,KCl,,,,bottle A\n',
    'Morning,a_20260923_1.txt,0,272.15,measurement,44.25,2.71,NaCl,28.39,0.5,,bath -1 C\n',
    'Morning,bad_20260923_2.txt,0,298.15,measurement,45.32,2.646,NaCl,28.39,0.5,x,excluded: fault\n',
    'Evening,b_20260923_3.txt,150,295.0,measurement,30.0,,NaCl,28.39,0.5,,vessel\n',
]


def _config(tmp_path, name='zAnalysis20260923.csv'):
    path = tmp_path / name
    path.write_text(HEADER + ''.join(ROWS))
    return path


def _assert_intact(path, n=4):
    df = pd.read_csv(path)
    assert len(df) == n
    assert 'Z_Ohm' in df.columns
    bad = df[df.filename == 'bad_20260923_2.txt'].iloc[0]
    assert bad.exclude == 'x' and bad.notes == 'excluded: fault' and bad.Z_Ohm == 45.32
    assert set(df.group_name) == {'Morning', 'Evening'}
    assert df.set_index('filename').notes['std.txt'] == 'bottle A'
    meas = [m for g in json.load(open(path.with_suffix('.json')))['calibration_groups']
            for m in g['measurements']]
    assert [m['filename'] for m in meas if m.get('exclude')] == ['bad_20260923_2.txt']
    return df


def test_update_config_changes_only_named_fields(tmp_path):
    path = _config(tmp_path)
    assert cs.update_config(path, {'b_20260923_3.txt': {'conductivity_Sm': 4.1}},
                            add_missing=False) == (1, 0)
    df = _assert_intact(path)
    assert df.set_index('filename').conductivity_Sm['b_20260923_3.txt'] == 4.1
    # CSV is newer, so auto-harmonize at load syncs CSV -> JSON
    assert os.stat(path).st_mtime_ns >= os.stat(path.with_suffix('.json')).st_mtime_ns


def test_json_round_trip_is_lossless(tmp_path):
    from harmonize_config import harmonize_config
    path = _config(tmp_path)
    json_path = harmonize_config(path, force=True, verbose=False)
    harmonize_config(json_path, force=True, verbose=False)
    df = _assert_intact(path)
    # measurement sigma survives JSON -> CSV (the GUI restores it)
    assert df.set_index('filename').conductivity_Sm['a_20260923_1.txt'] == 2.71


def test_headless_writeback_csv_and_json_inputs(tmp_path):
    from harmonize_config import harmonize_config
    from headless_analysis import _update_config_with_conductivities
    results = [{'type': 'measurement', 'filename': 'b_20260923_3.txt', 'conductivity_Sm': 4.2}]
    path = _config(tmp_path)
    _update_config_with_conductivities(AnalysisConfig(str(path)), results, tmp_path)
    _assert_intact(path)

    json_path = harmonize_config(path, force=True, verbose=False)
    path.unlink()
    _update_config_with_conductivities(AnalysisConfig(str(json_path)), results, tmp_path)
    df = _assert_intact(path)
    assert df.set_index('filename').conductivity_Sm['b_20260923_3.txt'] == 4.2


def test_borrowed_standards_replace_previous_and_keep_rows(tmp_path):
    from calibration_sharing import write_borrowed, BORROWED_TAG
    path = _config(tmp_path)
    std = {'filename': 'src_std.txt', 'conductivity_Sm': 8.0,
           'notes': f'{BORROWED_TAG}20260924'}
    write_borrowed(str(path), [std])
    write_borrowed(str(path), [dict(std, filename='src_std2.txt')])
    df = _assert_intact(path, n=6)
    borrowed = df[df.notes.str.startswith(BORROWED_TAG, na=False)]
    assert list(borrowed.filename) == ['src_std2.txt', 'src_std2.txt']
    assert set(borrowed.group_name) == {'Morning', 'Evening'}
    assert set(borrowed.type) == {'standard'}


def test_gui_save_merges_without_erasing(tmp_path):
    from hipoz_data_selector_gui import DataSelector
    path = _config(tmp_path)
    files =['/x/20260923/std.txt', 'a_20260923_1.txt', 'b_20260923_3.txt',
             'new_20260923_4.txt']
    cols = ['Comp', 'w (ppt)', 'w (molal)', 'P (MPa)', 'T (K)', 'Z (Ohm)',
            'Z± (Ohm)', 'σ (S/m)', 'σ± (S/m)']
    data = pd.DataFrame([
        ['KCl', np.nan, np.nan, 2.0, 301.0, 14.98, 0.1, 8.0, 0.2],
        # bath row: GUI shows the vessel header (2.3 MPa, 301 K)
        ['NaCl', 28.39, '0.5', 2.3, 301.45, 44.30, 0.2, 2.708, 0.1],
        ['NaCl', 28.39, '0.5', 150.0, 295.0, 30.1, 0.2, 3.98, 0.1],
        ['NaCl', 28.39, '0.5', 100.0, 296.0, 31.0, 0.2, np.nan, np.nan],
    ], columns=cols)
    fake = SimpleNamespace(
        analysis_config=SimpleNamespace(config_path=str(path)),
        config_file_paths={}, timeseries=SimpleNamespace(filenames=files), data=data,
        standard_mask=[True, False, False, False],
        associated_mask=[False, True, True, False])
    fake._merge_into_config = lambda p, d: DataSelector._merge_into_config(fake, p, d)
    fake._gui_row_fields = DataSelector._gui_row_fields
    DataSelector.save_gui_state_to_config(fake)

    df = _assert_intact(path, n=5).set_index('filename')
    assert (df.P_MPa['a_20260923_1.txt'], df.T_K['a_20260923_1.txt']) == (0, 272.15)
    assert df.conductivity_Sm['a_20260923_1.txt'] == 2.708
    assert df.Z_Ohm['b_20260923_3.txt'] == 30.1 and df.notes['b_20260923_3.txt'] == 'vessel'
    assert df.group_name['new_20260923_4.txt'] == 'Morning'
    assert df.P_MPa['new_20260923_4.txt'] == 100.0


def test_stale_json_newer_than_csv_cannot_erase_csv(tmp_path):
    """A JSON written by older code (no extra columns, a row missing, the old
    'calibrations' key) that is newer than the CSV: the load-time sync
    (JSON -> CSV) keeps the CSV's extra columns and rows."""
    from harmonize_config import harmonize_config
    path = _config(tmp_path)
    stale = {'calibrations': [{'name': 'Morning', 'standards': [
        {'filename': 'std.txt', 'P_MPa': 0, 'T_K': 298.15, 'conductivity_Sm': 8.0,
         'comp': 'KCl', 'notes': 'bottle A'}], 'measurements': [
        {'filename': 'a_20260923_1.txt', 'P_MPa': 0, 'T_K': 272.15, 'comp': 'NaCl',
         'w_ppt': 28.39, 'w_molal': 0.5, 'notes': 'bath -1 C'},
        'bad_20260923_2.txt']}], 'description': 'old'}
    json_path = path.with_suffix('.json')
    json_path.write_text(json.dumps(stale))
    os.utime(json_path, (os.stat(path).st_atime + 10, os.stat(path).st_mtime + 10))
    harmonize_config(path, force=True, verbose=False, auto_detect=True)
    df = pd.read_csv(path).set_index('filename')
    assert len(df) == 4 and df.Z_Ohm['bad_20260923_2.txt'] == 45.32
    assert df.notes['b_20260923_3.txt'] == 'vessel'
    # blank JSON fields do not blank the CSV: the exclusion survives
    assert df.exclude['bad_20260923_2.txt'] == 'x'
    assert df.notes['bad_20260923_2.txt'] == 'excluded: fault'

    # CSV -> JSON keeps the JSON's top-level keys; rows follow the CSV
    harmonize_config(path, force=True, verbose=False)
    data = json.load(open(json_path))
    assert data['description'] == 'old'
    assert sum(len(g['standards']) + len(g['measurements'])
               for g in data['calibration_groups']) == 4
