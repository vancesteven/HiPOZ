"""KCl standard bottle ppt values are g per kg water (Steve, 2026-11-06)."""
import pytest

from gamryTools import kcl_std_molal, wKCl_ppt, get_w_from_descrip


def test_80mS_bottle_molality_is_per_kg_water():
    assert kcl_std_molal(52.168) == pytest.approx(0.69977, rel=1e-4)


def test_descrip_label_uses_water_basis():
    comp, w_ppt, w_molal = get_w_from_descrip('KCl standard', lbl_uScm=80000)
    assert (comp, w_ppt) == ('KCl', 52.168)
    assert w_molal == pytest.approx(0.69977, rel=1e-4)


def test_kcl_bottle_ppt_round_trips():
    from gamryTools import ppt_to_molal, molal_to_ppt
    for w_ppt in wKCl_ppt.values():
        m = ppt_to_molal('KCl', w_ppt)
        assert m == pytest.approx(kcl_std_molal(w_ppt), rel=1e-12)
        assert molal_to_ppt('KCl', m) == w_ppt
    # The GUI's 74.55 g/mol and rounded molalities still map to the bottle
    assert ppt_to_molal('KCl', 52.168, 74.55) == pytest.approx(0.69977, rel=1e-4)
    assert molal_to_ppt('KCl', 0.699772, 74.55) == 52.168


def test_non_bottle_ppt_stays_per_kg_solution():
    from gamryTools import ppt_to_molal, molal_to_ppt
    from PlanetProfile.Thermodynamics.MgSO4.MgSO4Props import Ppt2molal, Molal2ppt
    assert ppt_to_molal('KCl', 30.0) == Ppt2molal(30.0, 74.55)
    assert ppt_to_molal('NaCl', 52.168) == Ppt2molal(52.168, 58.44)
    assert molal_to_ppt('NaCl', 0.5) == Molal2ppt(0.5, 58.44)


def test_gui_fills_kcl_molality_on_water_basis():
    """DataSelector.convert_loaded_concentrations, used when a config has ppt only."""
    from types import SimpleNamespace
    import pandas as pd
    from hipoz_data_selector_gui import DataSelector
    data = pd.DataFrame({'Comp': ['KCl', 'NaCl', 'KCl'],
                         'w (ppt)': [52.168, 28.39, None],
                         'w (molal)': [None, None, 0.699772]}, dtype=object)
    fake = SimpleNamespace(data=data, refresh_table=lambda: None)
    DataSelector.convert_loaded_concentrations(fake)
    assert data.at[0, 'w (molal)'] == pytest.approx(0.69977, rel=1e-4)
    assert data.at[1, 'w (molal)'] == pytest.approx(0.5, rel=1e-3)
    assert data.at[2, 'w (ppt)'] == 52.168
