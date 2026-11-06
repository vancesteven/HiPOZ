"""cortes_data_processing.load_cortes_data: config naming and exclusions."""
import pandas as pd

import cortes_data_processing as cdp

COLUMNS = ['filename', 'P_MPa', 'T_K', 'type', 'conductivity_Sm', 'comp',
           'w_molal', 'exclude']


def _write(tmp_path, folder, name, rows):
    d = tmp_path / 'data' / folder
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(d / name, index=False)


def test_loads_date_and_folder_named_configs_and_drops_excluded(tmp_path, monkeypatch):
    # Aug 18 style: zAnalysis<date>.csv
    _write(tmp_path, '20250818Cortes', 'zAnalysis20250818.csv', [
        ['a.txt', 100, 292, 'measurement', 4.9, 'NaCl', 0.5, False],
        ['std.txt', 0.1, 292, 'standard', 8.0, None, None, False],
    ])
    # Aug 19 style (--cal-map): zAnalysis<folder>.csv, with a flagged row
    _write(tmp_path, '20250819Cortes', 'zAnalysis20250819Cortes.csv', [
        ['b.txt', 200, 274, 'measurement', 3.4, 'NaCl', 0.5, False],
        ['c.txt', 150, 274, 'measurement', 3.3, 'NaCl', 0.5, True],
        ['d.txt', 150, 274, 'measurement', None, 'NaCl', 0.5, False],
    ])
    monkeypatch.chdir(tmp_path)
    df = cdp.load_cortes_data(['20250818Cortes', '20250819Cortes', '20250820Cortes'])
    assert sorted(df['filename']) == ['a.txt', 'b.txt']


def test_config_without_exclude_column_still_loads(tmp_path, monkeypatch):
    d = tmp_path / 'data' / '20250813Cortes'
    d.mkdir(parents=True)
    pd.DataFrame([['a.txt', 1, 292, 'measurement', 4.5, 'NaCl', 0.5]],
                 columns=COLUMNS[:-1]).to_csv(d / 'zAnalysis20250813.csv', index=False)
    monkeypatch.chdir(tmp_path)
    df = cdp.load_cortes_data(['20250813Cortes'])
    assert list(df['filename']) == ['a.txt']


def test_low_pressure_readings_are_assessed_at_ambient(tmp_path, monkeypatch):
    _write(tmp_path, '20250813Cortes', 'zAnalysis20250813.csv', [
        ['a.txt', 2.3, 293, 'measurement', 4.7, 'NaCl', 0.5, False],
        ['b.txt', 9.9, 293, 'measurement', 4.8, 'NaCl', 0.5, False],
        ['c.txt', 10.0, 293, 'measurement', 4.9, 'NaCl', 0.5, False],
        ['d.txt', 48.7, 293, 'measurement', 4.9, 'NaCl', 0.5, False],
    ])
    monkeypatch.chdir(tmp_path)
    df = cdp.load_cortes_data(['20250813Cortes']).set_index('filename')
    assert list(df['P_MPa']) == [0.1, 0.1, 10.0, 48.7]
    assert list(df['P_MPa_raw']) == [2.3, 9.9, 10.0, 48.7]
    # Gauge-offset replicates now average into one ambient point
    avg = cdp.average_replicates(df.reset_index().assign(w_ppt=28.39))
    assert (avg['P_MPa'] == 0.1).sum() == 1
