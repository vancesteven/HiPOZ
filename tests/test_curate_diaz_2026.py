"""curate_diaz_2026: excluded rows are flagged, annotated once, and skipped."""
import pandas as pd

import curate_diaz_2026 as cd
from analysis_config import AnalysisConfig

COLUMNS = ['group_name', 'filename', 'P_MPa', 'T_K', 'type', 'conductivity_Sm',
           'comp', 'w_ppt', 'w_molal', 'exclude', 'notes']
RULES = cd.EXCLUSIONS['20260923']


def _config(tmp_path):
    d = tmp_path / 'data' / '20260923'
    d.mkdir(parents=True)
    rows = [['G', f'Default_20260923_{t}_P_2_T_295.txt', 0, 298.15, typ, s,
             'NaCl', 28.39, 0.5, None, 'Z. Diaz analysis']
            for t, typ, s in [('111954', 'standard', 8.0),
                              ('173141', 'measurement', None),
                              ('174008', 'measurement', None),
                              ('174102', 'measurement', None),
                              ('174155', 'measurement', None)]]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(d / 'zAnalysis20260923.csv', index=False)


def test_end_of_trial2_excluded_once(tmp_path):
    _config(tmp_path)
    cfg, df, hits = cd.curate(str(tmp_path), '20260923', RULES)
    assert hits == 3
    assert list(df['exclude'].fillna('')) == ['', '', 'x', 'x', 'x']
    df.to_csv(cfg, index=False)
    _, df2, _ = cd.curate(str(tmp_path), '20260923', RULES)
    assert df2['notes'].str.count('end of Trial 2').tolist() == [0, 0, 1, 1, 1]

    df2.to_csv(cfg, index=False)
    kept = [m if isinstance(m, str) else m['filename']
            for m in AnalysisConfig(cfg).calibration_groups[0]['measurements']]
    assert kept == ['Default_20260923_173141_P_2_T_295.txt']


def test_whole_oct9_run_excluded(tmp_path):
    d = tmp_path / 'data' / '20261009'
    d.mkdir(parents=True)
    pd.DataFrame([['G', 'Default_20261009_110230_P_3_T_296.txt', 0, 295.86,
                   'measurement', None, 'NaCl', 50.021, 0.901, None, None]],
                 columns=COLUMNS).to_csv(d / 'zAnalysis20261009.csv', index=False)
    _, df, hits = cd.curate(str(tmp_path), '20261009', cd.EXCLUSIONS['20261009'])
    assert hits == 1 and df['notes'][0].startswith('excluded: sigma ~20% below')


def test_headless_writeback_keeps_excluded_rows_and_extra_columns(tmp_path):
    from headless_analysis import _update_config_with_conductivities
    _config(tmp_path)
    cfg, df, _ = cd.curate(str(tmp_path), '20260923', RULES)
    df.insert(5, 'Z_Ohm', [14.99, 23.6, 45.32, 47.83, 46.64])
    df.to_csv(cfg, index=False)
    ac = AnalysisConfig(cfg)
    results = [{'type': 'measurement', 'conductivity_Sm': 5.0794,
                'filename': 'Default_20260923_173141_P_2_T_295.txt'}]
    _update_config_with_conductivities(ac, results, tmp_path)
    out = pd.read_csv(cfg)
    assert len(out) == 5 and 'Z_Ohm' in out.columns
    assert list(out['exclude'].fillna('')) == ['', '', 'x', 'x', 'x']
    assert out['conductivity_Sm'][1] == 5.0794
    assert out['notes'].str.contains('end of Trial 2').sum() == 3
    import json
    meas = json.load(open(cfg.replace('.csv', '.json')))['calibration_groups'][0]['measurements']
    assert [m.get('exclude', False) for m in meas] == [False, True, True, True]


def test_harmonize_round_trip_keeps_extra_columns(tmp_path):
    from harmonize_config import harmonize_config
    _config(tmp_path)
    cfg, df, _ = cd.curate(str(tmp_path), '20260923', RULES)
    df.insert(5, 'Z_Ohm', [14.99, 23.6, 45.32, 47.83, 46.64])
    df.to_csv(cfg, index=False)
    from pathlib import Path
    json_path = harmonize_config(Path(cfg), force=True, verbose=False)
    harmonize_config(json_path, force=True, verbose=False)   # JSON -> CSV
    out = pd.read_csv(cfg)
    assert list(out['Z_Ohm']) == [14.99, 23.6, 45.32, 47.83, 46.64]
    assert list(out['exclude'].fillna('')) == ['', '', 'x', 'x', 'x']
    groups = AnalysisConfig(str(json_path)).calibration_groups
    assert len(groups[0]['measurements']) == 4
