"""Supplement tables carry the source values verbatim (no rounding, no loss)."""
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'cortes2026'))
import supplement_tables as stb  # noqa: E402

SUP = REPO / 'cortes2026' / 'supplement'


def _untex(s):
    s = s.strip()
    if s.startswith(r'\texttt{') and s.endswith('}'):
        s = s[len(r'\texttt{'):-1].replace(r'\allowbreak{}', '')
    for a, b in ((r'\textbackslash{}', '\\'), (r'\&', '&'), (r'\%', '%'), (r'\$', '$'), (r'\#', '#'),
                 (r'\_', '_'), (r'\{', '{'), (r'\}', '}'), (r'\textasciitilde{}', '~'),
                 (r'\textasciicircum{}', '^'), (r'\textmu{}', 'µ')):
        s = s.replace(a, b)
    return s


def _table_rows(name, label):
    text = (SUP / f'{name}_tables.tex').read_text()
    block = text.split(r'\label{' + label + '}')[1].split(r'\end{longtable}')[0]
    body = block.split(r'\endfoot')[1]
    rows = [r for r in body.split(r' \\') if r.strip()]
    return [[_untex(c) for c in re.split(r'(?<!\\)&', r)] for r in rows]


def _src(path, **kw):
    return pd.read_csv(REPO / path, dtype=str, keep_default_na=False, **kw)


@pytest.fixture(scope='module', autouse=True)
def generated():
    stb.main()


@pytest.mark.parametrize('name,label,paths', [
    ('cortes', 'tab:cortes-sweeps', stb.CORTES_CONFIGS),
    ('diaz', 'tab:diaz-1bar', stb.DIAZ_1BAR_CONFIGS),
    ('garrity', 'tab:garrity-gamry', [stb.GARRITY_CONFIG]),
])
def test_config_sweeps_verbatim(name, label, paths):
    rows = _table_rows(name, label)
    src = pd.concat([_src(p) for p in paths], ignore_index=True).fillna('')
    assert len(rows) == len(src)
    for r, (_, s) in zip(rows, src.iterrows()):
        assert r[:9] == [s.filename, s.type, s.P_MPa, s.T_K, s.get('Z_Ohm', ''), s.get('Z_unc_Ohm', ''),
                         s.conductivity_Sm, s.comp, s.w_molal]
        excluded = s.exclude.lower() in ('x', 'true', '1')
        assert r[9].startswith('excluded') == excluded


def test_diaz_hp_sweeps_verbatim():
    rows = _table_rows('diaz', 'tab:diaz-sweeps')
    src = _src(stb.DIAZ_HP_SWEEPS)
    assert len(rows) == len(src)
    for r, (_, s) in zip(rows, src.iterrows()):
        assert r == [s.filename, s.run, s.P_MPa, s.T_K, s.R, s.R_unc, s.K_cell, s.sigma_Sm, s.excluded]


def test_benchtop_verbatim():
    rows = _table_rows('cortes', 'tab:cortes-benchtop')
    src = _src(stb.BENCHTOP, comment='#')
    assert len(rows) == len(src)
    for r, (_, s) in zip(rows, src.iterrows()):
        assert r == [s.compound, s.concentration_molal, s.temperature_C, s.conductivity_Sm,
                     s.replicate, s.notes]


def test_garrity_probe_verbatim():
    rows = _table_rows('garrity', 'tab:garrity-probe')
    src = _src(stb.GARRITY_PROBE, comment='#')
    assert len(rows) == len(src) == 138
    for r, (_, s) in zip(rows, src.iterrows()):
        assert r[6] == s.value and r[3] == s.concentration and r[2] == s.solute


def test_step_tables_keep_every_step_and_flag():
    st = _src(stb.CORTES_STEPS)
    rows = _table_rows('cortes', 'tab:cortes-steps')
    assert len(rows) == len(st) == 71
    assert sum(1 for r in rows if r[-1]) == (st.excluded != '').sum() == 12
