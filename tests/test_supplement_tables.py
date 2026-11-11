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
    s = s.strip().replace(r'\allowbreak{}', '')
    if s.startswith(r'\texttt{') and s.endswith('}'):
        s = s[len(r'\texttt{'):-1]
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


def _key(name, label, title):
    """Decode a table's code key, written after the table: {code: text}."""
    text = (SUP / f'{name}_tables.tex').read_text()
    after = text.split(r'\label{' + label + '}')[1].split(r'\end{longtable}', 1)[1]
    after = after.split(r'\begin{longtable}', 1)[0]
    m = re.search(r'\\textit\{' + re.escape(title) + r':\} (.*?)\.\\par\\endgroup', after, re.S)
    assert m, f'no key "{title}" after {label}'
    out = {}
    for part in m.group(1).split(r' \quad '):
        c, v = re.match(r'\\textbf\{(.*?)\} (.*)', part, re.S).groups()
        out[c] = _untex(v)
    return out


def _decode(code, key):
    return key[code] if code else ''


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
        flags = _key(name, label, 'Flags') if any(row[9] for row in rows) else {}
        assert _decode(r[9], flags).startswith('excluded') == excluded


def test_diaz_hp_sweeps_verbatim():
    rows = _table_rows('diaz', 'tab:diaz-sweeps')
    src = _src(stb.DIAZ_HP_SWEEPS)
    runs = _key('diaz', 'tab:diaz-sweeps', 'Runs')
    flags = _key('diaz', 'tab:diaz-sweeps', 'Flags')
    assert len(rows) == len(src)
    for r, (_, s) in zip(rows, src.iterrows()):
        assert r[:1] + r[2:8] == [s.filename, s.P_MPa, s.T_K, s.R, s.R_unc, s.K_cell, s.sigma_Sm]
        assert _decode(r[1], runs) == s.run
        assert _decode(r[8], flags) == s.excluded


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
    notes = _key('garrity', 'tab:garrity-probe', 'Notes')
    assert len(rows) == len(src) == 138
    for r, (_, s) in zip(rows, src.iterrows()):
        assert r[:9] == [s.page, s.date, s.solute, s.concentration, s.g_MgSO4_per_10mL, s.reading,
                         s.value, s.reading_units, s.T_C or s.T_C_written]
        assert _decode(r[9], notes) == s.notes


def test_step_tables_keep_every_step_and_flag():
    st = _src(stb.CORTES_STEPS)
    rows = _table_rows('cortes', 'tab:cortes-steps')
    key = _key('cortes', 'tab:cortes-steps', 'Left out of the figure')
    assert len(rows) == len(st) == 71
    assert [_decode(r[-1], key) for r in rows] == list(st.excluded)
    assert sum(1 for r in rows if r[-1]) == 12
