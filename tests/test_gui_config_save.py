"""The GUI's config save/restore must not change what the user did not touch.

Real DataSelector instances (offscreen) on a synthetic data folder:
open + close without action leaves the CSV byte-identical; folder-suffixed
configs are found; only explicit actions change type / P / T / sigma."""
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from gamryTools import TimeSeries
import hipoz_data_selector_gui as gui_mod
from hipoz_data_selector_gui import DataSelector

matplotlib.rcParams['text.usetex'] = False   # no TeX needed to exercise the save path
app = QApplication.instance() or QApplication(sys.argv)

HEADER = ('group_name,filename,P_MPa,T_K,type,Z_Ohm,conductivity_Sm,comp,w_ppt,'
          'w_molal,exclude,notes\n')
FILES = ['Std_20250819_100000_P_3_T_298.txt',      # standard (own folder)
         'Default_20250819_110000_P_3_T_274.txt',  # bath row, P=0 in config
         'Default_20250819_110100_P_150_T_295.txt',
         'Default_20250819_110200_P_150_T_295.txt',  # excluded
         'Default_20250819_110300_P_100_T_290.txt']  # not in config
ROWS = [
    'G,Std_20250819_100000_P_3_T_298.txt,0.1,298.15,standard,14.987654321,8.0123456789,KCl,,,,bottle\n',
    'G,Std_20250818_090000_borrowed.txt,,,standard,,8.0,KCl,,,,borrowed_from:20250818Cortes\n',
    'G,Default_20250819_110000_P_3_T_274.txt,0,272.15,measurement,44.251234567,2.78739036912,NaCl,28.39,0.5,,bath -1 C\n',
    'G,Default_20250819_110100_P_150_T_295.txt,150,295.0,measurement,30.123456789,3.98765432101,NaCl,28.39,0.5,,vessel\n',
    'G,Default_20250819_110200_P_150_T_295.txt,150,295.0,measurement,31.5,,NaCl,28.39,0.5,x,excluded: fault\n',
]


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)            # hipoz_exports/ lands here
    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, 'information', staticmethod(lambda *a, **k: None))
    shown = []
    monkeypatch.setattr(DataSelector, 'show_config_status_dialog',
                        lambda self, *a: shown.append(a))
    return shown


def _folder(tmp_path, name='20250819Cortes', cfg='zAnalysis20250819Cortes.csv',
            header=HEADER, rows=ROWS):
    folder = tmp_path / 'data' / name
    (folder / 'ConductivityData_Default').mkdir(parents=True)
    if cfg:
        (folder / cfg).write_text(header + ''.join(rows))
    return folder


def _ts(folder, files=FILES):
    n = len(files)
    ts = TimeSeries()
    ts.filenames = [str(folder / 'ConductivityData_Default' / f) for f in files]
    ts.timestamps = [f"2025-08-19 {10 + i}:00:00" for i in range(n)]
    ts.Ts = [298.0, 273.9, 294.9, 294.9, 290.0][:n]       # file headers
    ts.Ps = [3, 3, 150, 150, 100][:n]
    ts.Rcalc_ohm = [14.9, 44.2, 30.1, 31.5, 31.0][:n]
    ts.percent_uncertainties = [0.5] * n
    ts.uncertainties = [0.1] * n
    ts.colors = ['C0'] * n
    ts.markers = ['o'] * n
    ts.conductivities_Sm = [None] * n
    ts.conductivities_unc_pct = [None] * n
    return ts


def _open(folder, **kw):
    return DataSelector(_ts(folder, **kw))


def test_open_close_without_action_leaves_csv_byte_identical(tmp_path):
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    before = cfg.read_bytes()
    ds = _open(folder)
    assert ds.standard_mask[0]            # config standard restored to the mask
    ds.close()                            # closeEvent saves
    assert cfg.read_bytes() == before
    assert sorted(p.name for p in folder.glob('zAnalysis*')) == [cfg.name]


def test_folder_suffixed_config_found_no_stray_file(tmp_path, _quiet):
    folder = _folder(tmp_path)
    ds = _open(folder)
    assert Path(ds.config_file_paths['20250819Cortes']).name == 'zAnalysis20250819Cortes.csv'
    ds.close()
    assert not (folder / 'zAnalysis20250819.csv').exists()
    assert _quiet == []                   # no "config created" dialog


def test_same_date_folders_do_not_cross_write(tmp_path):
    a = _folder(tmp_path, name='20250819', cfg='zAnalysis20250819.csv')
    b = _folder(tmp_path)
    before_a = (a / 'zAnalysis20250819.csv').read_bytes()
    ds = DataSelector(_ts(b))
    ds.edited_cells.add((FILES[2], 'P_MPa'))
    ds.data.iat[2, ds.data.columns.get_loc('P (MPa)')] = 77
    ds.save_gui_state_to_config()
    ds.close()
    assert (a / 'zAnalysis20250819.csv').read_bytes() == before_a
    df = pd.read_csv(b / 'zAnalysis20250819Cortes.csv').set_index('filename')
    assert df.P_MPa[FILES[2]] == 77


def test_config_found_when_analysis_config_given(tmp_path):
    from analysis_config import AnalysisConfig
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    ds = DataSelector(_ts(folder), analysis_config=AnalysisConfig(str(cfg)))
    assert Path(ds.config_file_paths['20250819Cortes']) == cfg
    ds.close()
    assert [p.name for p in folder.glob('zAnalysis*.csv')] == [cfg.name]


def test_standard_not_demoted_when_mask_unset(tmp_path):
    """Even if the restore never set the mask, a save keeps type=standard."""
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    ds = _open(folder)
    ds.standard_mask[:] = False
    ds.save_gui_state_to_config()
    ds.close()
    df = pd.read_csv(cfg).set_index('filename')
    assert df.type[FILES[0]] == 'standard'
    assert df.type['Std_20250818_090000_borrowed.txt'] == 'standard'


def test_mark_as_standard_and_associate_change_only_intended_fields(tmp_path):
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    ds = _open(folder)
    before = pd.read_csv(cfg, dtype=str, keep_default_na=False).set_index('filename')

    # mark the (restored) standard again, then associate row 2
    ds.table.selectRow(0)
    ds.mark_as_standard()
    assert ds.explicit_types == {FILES[0]: 'standard'}
    assert ds.current_std == pytest.approx(8.0123456789 * 14.987654321, rel=1e-3)
    ds.table.clearSelection()
    ds.table.selectRow(4)                  # the file that is not in the config
    ds.associate_measurements()
    ds.close()

    after = pd.read_csv(cfg, dtype=str, keep_default_na=False).set_index('filename')
    changed = [(f, c) for f in before.index for c in before.columns
               if before.at[f, c] != after.at[f, c]]
    assert changed == []                   # existing rows untouched (same values)
    new = after.loc[FILES[4]]
    assert new.type == 'measurement' and float(new.conductivity_Sm) > 0
    assert float(new.P_MPa) == 100 and float(new.T_K) == 290   # new row: GUI values
    assert len(after) == len(before) + 1


def test_mark_as_standard_promotes_a_measurement(tmp_path):
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    ds = _open(folder)
    ds.table.selectRow(2)
    ds.mark_as_standard()
    ds.close()
    df = pd.read_csv(cfg, dtype=str, keep_default_na=False).set_index('filename')
    assert df.type[FILES[2]] == 'standard'
    assert df.type[FILES[1]] == 'measurement' and df.type[FILES[0]] == 'standard'
    assert df.P_MPa[FILES[2]] == '150' and df.conductivity_Sm[FILES[2]] == '3.98765432101'


def test_user_edit_of_P_cell_is_saved_and_others_are_not(tmp_path):
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    ds = _open(folder)
    col_P = list(ds.data.columns).index('P (MPa)')
    ds.table.item(2, col_P).setText('120')            # fires on_table_item_changed
    ds.close()
    df = pd.read_csv(cfg, dtype=str, keep_default_na=False).set_index('filename')
    assert float(df.P_MPa[FILES[2]]) == 120
    assert df.P_MPa[FILES[1]] == '0' and df.T_K[FILES[1]] == '272.15'   # bath row kept
    assert df.T_K[FILES[2]] == '295.0'
    assert df.conductivity_Sm[FILES[2]] == '3.98765432101'
    assert df.exclude[FILES[3]] == 'x' and df.notes[FILES[3]] == 'excluded: fault'


def test_sigma_rewritten_only_when_value_really_changes(tmp_path):
    folder = _folder(tmp_path)
    cfg = folder / 'zAnalysis20250819Cortes.csv'
    ds = _open(folder)
    col_S = list(ds.data.columns).index('σ (S/m)')
    ds.table.item(2, col_S).setText('3.9877')         # same at 4 s.f.: keep
    ds.close()
    df = pd.read_csv(cfg, dtype=str, keep_default_na=False).set_index('filename')
    assert df.conductivity_Sm[FILES[2]] == '3.98765432101'
    ds = _open(folder)
    ds.table.item(2, col_S).setText('4.5')            # real change
    ds.close()
    df = pd.read_csv(cfg, dtype=str, keep_default_na=False).set_index('filename')
    assert float(df.conductivity_Sm[FILES[2]]) == 4.5


def test_folder_without_config_creates_one_named_for_folder(tmp_path):
    folder = _folder(tmp_path, name='20250820Cortes', cfg=None)
    files = [f.replace('20250819', '20250820') for f in FILES]
    ds = DataSelector(_ts(folder, files=files))
    ds.close()
    assert [p.name for p in folder.glob('zAnalysis*.csv')] == ['zAnalysis20250820Cortes.csv']
