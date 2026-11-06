"""McCleskey (2012) validity limit (I_eff = 0.7 mol/kg) and mixture speciation
in the Cortes paper figures (study_plots)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cortes_mccleskey as cm
import study_plots
from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHTOP = os.path.join(REPO, 'cortes2026', 'Cortes2026BenchtopData.csv')


@pytest.fixture(autouse=True)
def _no_tex():
    # study_plots imports plotting, which turns TeX on; containers lack latex
    plt.rcParams['text.usetex'] = False
    yield
    plt.close('all')


@pytest.fixture
def benchtop():
    return study_plots.load_study_data(BENCHTOP)


def test_chloride_limit_is_nominal():
    # No WATEQ4F ion pairs for NaCl/KCl, so I_eff = m
    assert cm.get_mccleskey_limit('NaCl') == pytest.approx(cm.MCCLESKEY_I_MAX)
    assert cm.get_mccleskey_limit('KCl') == pytest.approx(cm.MCCLESKEY_I_MAX)


def test_sulfate_limits_exceed_nominal_ionic_strength_limit():
    # Ion pairing lowers I_eff, so the limit lies above 0.7 / (nominal I per mol)
    nominal_I_per_mol = {
        'Na2SO4': 3.0, 'MgSO4': 4.0,
        'Na2SO4:KCl_1:1': 3.0 + 1.0, 'Na2SO4:KCl_1:2': 3.0 + 2.0,
        'Na2SO4:KCl_2:1': 3.0 + 0.5,
        'NaCl:MgSO4_1:1': 1.0 + 4.0, 'NaCl:MgSO4_1:2': 1.0 + 8.0,
        'NaCl:MgSO4_2:1': 1.0 + 2.0,
    }
    for compound, per_mol in nominal_I_per_mol.items():
        limit = cm.get_mccleskey_limit(compound)
        assert limit > cm.MCCLESKEY_I_MAX / per_mol, compound
        # and the nominal ionic strength at the limit stays physically close
        # (pairing removes at most about half of I for these salts)
        assert limit * per_mol < 2.5 * cm.MCCLESKEY_I_MAX, compound


def test_unlisted_compound_has_no_limit():
    assert cm.get_mccleskey_limit('Glycine') is None


@pytest.mark.parametrize('compound', ['Na2SO4:KCl_1:1', 'NaCl:MgSO4_1:1'])
def test_mixture_figures_request_speciation(monkeypatch, benchtop, tmp_path, compound):
    """Regression: mixture branches in study_plots dropped speciation=True."""
    calls = []
    real = study_plots.compute_mccleskey_model

    def spy(*args, **kwargs):
        calls.append(kwargs)
        kwargs = dict(kwargs, speciation=False)  # no Reaktoro needed here
        return real(*args, **kwargs)

    monkeypatch.setattr(study_plots, 'compute_mccleskey_model', spy)
    spec = cm.get_mixture_ion_spec(compound)
    study_plots.plot_study_concentration(
        benchtop, compound, str(tmp_path / 'c.pdf'), show_delta=True,
        ion_spec=spec, speciation=True)
    study_plots.plot_study_temperature(
        benchtop, compound, str(tmp_path / 't.pdf'), show_delta=True,
        ion_spec=spec, speciation=True)
    assert len(calls) == 2
    for kw in calls:
        assert kw['speciation'] is True
        assert kw['ion_spec'] == spec


def test_concentration_plot_marks_limit_on_both_panels(benchtop, tmp_path):
    limit = cm.get_mccleskey_limit('Na2SO4:KCl_1:1')
    fig = study_plots.plot_study_concentration(
        benchtop, 'Na2SO4:KCl_1:1', str(tmp_path / 'c.pdf'), show_delta=True,
        ion_spec=cm.get_mixture_ion_spec('Na2SO4:KCl_1:1'),
        mccleskey_limit=limit)
    for ax in fig.axes[:2]:
        lo, hi = ax.get_xlim()
        # every data point lies beyond this limit; the line must stay visible
        assert lo < limit < hi
        vlines = [l for l in ax.lines
                  if len(l.get_xdata()) == 2
                  and np.allclose(l.get_xdata(), [limit, limit])]
        assert vlines, 'no limit line'
    texts = ' '.join(t.get_text() for t in fig.axes[0].texts)
    assert 'MC12 limit' in texts


def test_temperature_plot_stars_curves_beyond_limit(benchtop, tmp_path):
    fig = study_plots.plot_study_temperature(
        benchtop, 'NaCl', str(tmp_path / 't.pdf'), show_delta=True,
        mccleskey_limit=cm.get_mccleskey_limit('NaCl'))
    texts = [t.get_text() for t in fig.axes[0].texts]
    starred = [t for t in texts if t.endswith('*')]
    # NaCl data: 0.5054 inside, 0.7616 / 1.02 / 1.546 beyond 0.7
    assert sorted(starred) == ['0.7616*', '1.0200*', '1.5460*']
    assert any('beyond MC12 range' in t for t in texts)


def test_ionic_strength_at_chloride_limit():
    # Sanity: the model's own I at the NaCl limit equals 0.7
    ions = {'Na_p1': {'mols': np.array([0.7])}, 'Cl_m1': {'mols': np.array([0.7])}}
    out = elecCondMcCleskey2012(20.0, ions)
    assert out['Na_p1']['I'][0] == pytest.approx(0.7)
