"""Delta panels must state the sign convention: (data - model)/model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotting


def test_delta_label_states_sign_convention():
    assert r'\sigma_\mathrm{data} - \sigma_\mathrm{model}' in plotting.DELTA_LABEL
    assert plotting.DELTA_LABEL.index('data') < plotting.DELTA_LABEL.index('model')


def test_concentration_plot_uses_delta_label(tmp_path):
    plt.rcParams['text.usetex'] = False
    fig = plotting.plot_sigma_vs_concentration(
        [0.5, 1.0], [[4.47, 8.23]], temp_labels=['20 C'],
        model_data=[[4.14, 7.82]], show_delta=True,
        out_file=str(tmp_path / 'delta.png'))
    fig = fig if fig is not None else plt.gcf()
    assert fig.axes[1].get_ylabel() == plotting.DELTA_LABEL
    # Positive Delta = data above model
    y = fig.axes[1].lines[0].get_ydata()
    assert y[0] > 0
    plt.close('all')
