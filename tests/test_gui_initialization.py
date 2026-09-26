#!/usr/bin/env python
"""
Test GUI initialization and tab creation.

Tests the DataSelector GUI class initialization, tab structure,
and widget creation without launching the full application.
"""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import QApplication
from hipoz_data_selector_gui import DataSelector
from gamryTools import TimeSeries

# Initialize QApplication for testing (required for PyQt6 widgets)
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

def create_mock_timeseries():
    """Create a minimal TimeSeries object for testing GUI initialization."""
    ts = TimeSeries()

    # Mock data for 5 measurements
    n = 5
    ts.filenames = [f"test_file_{i}.txt" for i in range(n)]
    ts.timestamps = [f"2025-01-{i+10:02d} 12:00:00" for i in range(n)]
    ts.Ts = [298.15, 298.15, 298.15, 298.15, 298.15]  # 25°C
    ts.Ps = [10, 20, 30, 40, 50]  # MPa
    ts.Rcalc_ohm = [100.0, 95.0, 90.0, 85.0, 80.0]
    ts.percent_uncertainties = [2.0, 2.0, 2.0, 2.0, 2.0]
    ts.conductivities_Sm = [None, None, None, None, None]
    ts.conductivities_unc_pct = [None, None, None, None, None]

    return ts

def test_gui_initialization():
    """Test that GUI initializes without errors."""
    print("=== Test: GUI Initialization ===\n")

    ts = create_mock_timeseries()

    # Initialize GUI
    gui = DataSelector(ts, analysis_config=None)

    # Verify GUI was created
    assert gui is not None, "GUI object should not be None"
    assert gui.windowTitle() == 'Gamry Data', "Window title should be 'Gamry Data'"

    print("✓ GUI initialized successfully")
    print(f"  Window title: {gui.windowTitle()}")
    print(f"  Window size: {gui.width()}x{gui.height()}\n")

    return gui

def test_tab_structure(gui):
    """Test that all expected tabs are created."""
    print("=== Test: Tab Structure ===\n")

    # Check tab widget exists
    assert hasattr(gui, 'tabs'), "GUI should have a 'tabs' attribute"

    # Get expected tab names
    expected_tabs = ["Timeseries", "Bode & Nyquist", "S vs P"]

    # Count tabs
    tab_count = gui.tabs.count()
    assert tab_count == len(expected_tabs), \
        f"Expected {len(expected_tabs)} tabs, found {tab_count}"

    # Verify tab names
    actual_tabs = []
    for i in range(tab_count):
        tab_name = gui.tabs.tabText(i)
        actual_tabs.append(tab_name)
        print(f"  Tab {i}: {tab_name}")

    for expected in expected_tabs:
        assert expected in actual_tabs, f"Missing expected tab: {expected}"

    print(f"\n✓ All {tab_count} tabs present\n")
    return True

def test_data_table_creation(gui):
    """Test that data table is created and populated."""
    print("=== Test: Data Table Creation ===\n")

    # Check table exists
    assert hasattr(gui, 'table'), "GUI should have a 'table' attribute"

    # Check table dimensions
    row_count = gui.table.rowCount()
    col_count = gui.table.columnCount()

    assert row_count == 5, f"Expected 5 rows, found {row_count}"
    assert col_count > 0, "Table should have columns"

    print(f"  Table dimensions: {row_count} rows × {col_count} columns")

    # Check expected columns
    expected_columns = ['Filename', 'Calibration', 'Time', 'Comp', 'w (ppt)',
                        'w (molal)', 'T (K)', 'P (MPa)', 'Z (Ohm)', 'Z± (Ohm)',
                        'S (S/m)', 'S± (S/m)']

    actual_columns = []
    for i in range(col_count):
        col_name = gui.table.horizontalHeaderItem(i).text()
        actual_columns.append(col_name)

    print(f"  Columns: {', '.join(actual_columns)}")

    for expected in expected_columns:
        assert expected in actual_columns, f"Missing expected column: {expected}"

    print("\n✓ Data table created successfully\n")
    return True

def test_button_creation(gui):
    """Test that all control buttons are created."""
    print("=== Test: Button Creation ===\n")

    expected_buttons = [
        'btn_clear_selection',
        'btn_mark_standard',
        'btn_associate_measurements',
        'btn_bulk_edit',
        'btn_reload_csv',
        'btn_create_plots',
        'btn_export_plots'
    ]

    for btn_name in expected_buttons:
        assert hasattr(gui, btn_name), f"Missing button: {btn_name}"
        button = getattr(gui, btn_name)
        print(f"  ✓ {btn_name}: '{button.text()}'")

    print(f"\n✓ All {len(expected_buttons)} buttons created\n")
    return True

def test_plot_canvases(gui):
    """Test that plot canvases are created for each tab."""
    print("=== Test: Plot Canvas Creation ===\n")

    # Check timeseries canvas
    assert hasattr(gui, 'canvas'), "Missing timeseries canvas"
    assert hasattr(gui, 'figure'), "Missing timeseries figure"
    print("  ✓ Timeseries: canvas and figure")

    # Check Bode canvas
    assert hasattr(gui, 'bode_canvas'), "Missing Bode canvas"
    assert hasattr(gui, 'bode_figure'), "Missing Bode figure"
    print("  ✓ Bode: canvas and figure")

    # Check Nyquist canvas
    assert hasattr(gui, 'nyquist_canvas'), "Missing Nyquist canvas"
    assert hasattr(gui, 'nyquist_figure'), "Missing Nyquist figure"
    print("  ✓ Nyquist: canvas and figure")

    # Check S vs P canvas
    assert hasattr(gui, 'svp_canvas'), "Missing S vs P canvas"
    assert hasattr(gui, 'svp_figure'), "Missing S vs P figure"
    print("  ✓ S vs P: canvas and figure")

    print("\n✓ All plot canvases created\n")
    return True

def test_dataframe_structure(gui):
    """Test that internal DataFrame has correct structure."""
    print("=== Test: DataFrame Structure ===\n")

    assert hasattr(gui, 'data'), "GUI should have a 'data' DataFrame"

    df = gui.data
    print(f"  DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Verify row count matches mock data
    assert len(df) == 5, f"Expected 5 rows, found {len(df)}"

    # Verify required columns
    required_columns = ['Filename', 'Time', 'P (MPa)', 'T (K)', 'Z (Ohm)']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
        print(f"  ✓ Column present: {col}")

    # Verify filename data is populated
    assert df['Filename'].iloc[0] == 'test_file_0.txt', \
        "Filename data not populated correctly"

    # Verify pressure data
    assert df['P (MPa)'].iloc[0] == 10, "Pressure data not populated correctly"

    print("\n✓ DataFrame structure valid\n")
    return True

def test_mask_initialization(gui):
    """Test that standard and associated masks are initialized."""
    print("=== Test: Mask Initialization ===\n")

    assert hasattr(gui, 'standard_mask'), "Missing standard_mask"
    assert hasattr(gui, 'associated_mask'), "Missing associated_mask"

    # Check mask dimensions
    n = len(gui.data)
    assert len(gui.standard_mask) == n, \
        f"standard_mask length {len(gui.standard_mask)} != data length {n}"
    assert len(gui.associated_mask) == n, \
        f"associated_mask length {len(gui.associated_mask)} != data length {n}"

    # Check all masks are False initially
    assert not np.any(gui.standard_mask), "standard_mask should be all False initially"
    assert not np.any(gui.associated_mask), "associated_mask should be all False initially"

    print(f"  ✓ standard_mask: shape={gui.standard_mask.shape}, all False")
    print(f"  ✓ associated_mask: shape={gui.associated_mask.shape}, all False")
    print("\n✓ Masks initialized correctly\n")
    return True

def test_status_bar(gui):
    """Test that status bar is created."""
    print("=== Test: Status Bar ===\n")

    assert hasattr(gui, 'status_bar'), "Missing status_bar"

    # Check initial message
    status_message = gui.status_bar.currentMessage()
    assert status_message == "Ready", f"Expected 'Ready', got '{status_message}'"

    print(f"  ✓ Status bar message: '{status_message}'")
    print("\n✓ Status bar created\n")
    return True

if __name__ == '__main__':
    try:
        # Run all tests
        gui = test_gui_initialization()
        test_tab_structure(gui)
        test_data_table_creation(gui)
        test_button_creation(gui)
        test_plot_canvases(gui)
        test_dataframe_structure(gui)
        test_mask_initialization(gui)
        test_status_bar(gui)

        print("=" * 50)
        print("✓ ALL GUI INITIALIZATION TESTS PASSED")
        print("=" * 50)

        # Clean up
        gui.close()

    except AssertionError as e:
        print("\n" + "=" * 50)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 50)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)
