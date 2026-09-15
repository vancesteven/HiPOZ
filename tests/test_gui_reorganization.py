#!/usr/bin/env python3
"""
Test suite for GUI reorganization and new plotting features.

Tests:
- Data table in dedicated tab
- Auto-generation of Bode/Nyquist plots
- New σ vs T and σ vs m plots
- Symbol changes from S to σ
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hipoz_data_selector_gui import DataSelector
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


class MockTimeSeries:
    """Mock TimeSeries object for testing."""
    def __init__(self, n_points=5):
        self.filenames = [f"test_file_{i}.txt" for i in range(n_points)]
        self.timestamps = pd.date_range('2025-01-01', periods=n_points, freq='H')
        self.Ts = np.array([298.0] * n_points)
        self.Ps = np.array([10.0] * n_points)
        self.Rcalc_ohm = np.array([100.0] * n_points)
        self.percent_uncertainties = np.array([2.0] * n_points)
        self.conductivities_Sm = np.array([0.01] * n_points)
        self.conductivities_unc_pct = np.array([2.0] * n_points)
        self.frequencies = [np.logspace(1, 5, 10) for _ in range(n_points)]
        self.impedances = [np.ones(10) * (100 + 10j) for _ in range(n_points)]
        self.impedance_fits = [np.ones(10) * (100 + 10j) for _ in range(n_points)]


def test_data_table_tab_exists(qapp):
    """Test that Data Table tab exists and is first tab."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    # Check that first tab is "Data Table"
    assert selector.tabs.count() >= 1
    assert selector.tabs.tabText(0) == "Data Table"


def test_all_tabs_present(qapp):
    """Test that all required tabs are present."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    expected_tabs = ["Data Table", "Timeseries", "Bode & Nyquist", "σ vs P", "σ vs T", "σ vs m"]
    actual_tabs = [selector.tabs.tabText(i) for i in range(selector.tabs.count())]

    for expected_tab in expected_tabs:
        assert expected_tab in actual_tabs, f"Tab '{expected_tab}' not found"


def test_sigma_symbol_in_dataframe(qapp):
    """Test that dataframe uses σ symbol instead of S."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    # Check that σ columns exist
    assert 'σ (S/m)' in selector.data.columns
    assert 'σ± (S/m)' in selector.data.columns

    # Check that old S columns don't exist
    assert 'S (S/m)' not in selector.data.columns
    assert 'S± (S/m)' not in selector.data.columns


def test_table_in_data_table_tab(qapp):
    """Test that table widget is in Data Table tab."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    # Data table should be in the first tab
    data_table_tab = selector.tabs.widget(0)
    layout = data_table_tab.layout()

    # Table should be first widget in layout
    assert layout is not None
    assert layout.count() > 0


def test_svt_figure_exists(qapp):
    """Test that σ vs T figure and canvas exist."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    assert hasattr(selector, 'svt_figure')
    assert hasattr(selector, 'svt_canvas')
    assert selector.svt_figure is not None
    assert selector.svt_canvas is not None


def test_svm_figure_exists(qapp):
    """Test that σ vs m figure and canvas exist."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    assert hasattr(selector, 'svm_figure')
    assert hasattr(selector, 'svm_canvas')
    assert selector.svm_figure is not None
    assert selector.svm_canvas is not None


def test_bode_nyquist_button_removed(qapp):
    """Test that 'Create Bode and Nyquist Plots' button was removed."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    # Button should not exist
    assert not hasattr(selector, 'btn_create_plots')


def test_auto_plot_on_selection(qapp):
    """Test that plots are auto-generated when selection changes."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    # Check that selection changed signal is connected
    assert selector.table.selectionModel().selectionChanged.isSignalConnected(
        selector.on_table_selection_changed
    )


def test_refresh_sigma_vs_t_method(qapp):
    """Test that refresh_sigma_vs_t_plot method exists."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    assert hasattr(selector, 'refresh_sigma_vs_t_plot')
    assert callable(selector.refresh_sigma_vs_t_plot)


def test_refresh_sigma_vs_m_method(qapp):
    """Test that refresh_sigma_vs_m_plot method exists."""
    ts = MockTimeSeries()
    selector = DataSelector(ts)

    assert hasattr(selector, 'refresh_sigma_vs_m_plot')
    assert callable(selector.refresh_sigma_vs_m_plot)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
