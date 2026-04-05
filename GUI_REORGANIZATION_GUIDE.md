# HiPOZ GUI Reorganization Guide

## Overview

The HiPOZ data selector GUI has been reorganized to improve usability and provide better access to data visualization tools. This guide documents the changes and how to use the new features.

## Major Changes

### 1. Tab Reorganization

**Previous Structure:**
- Data table and plots side-by-side in horizontal split view
- Three tabs: Timeseries, Bode & Nyquist, S vs P

**New Structure:**
- **Six-tab interface with data table as first tab:**
  1. **Data Table** (NEW) - Full-window data view with editing controls
  2. **Timeseries** - Time series plots of impedance measurements
  3. **Bode & Nyquist** - Impedance spectroscopy plots
  4. **σ vs P** - Conductivity vs pressure plots (renamed from "S vs P")
  5. **σ vs T** (NEW) - Conductivity vs temperature plots
  6. **σ vs m** (NEW) - Conductivity vs molality plots

### 2. Symbol Changes: S → σ

All references to conductivity have been changed from "S" to the proper symbol "σ" (sigma):

- DataFrame columns: `'σ (S/m)'` and `'σ± (S/m)'`
- Plot labels: Use LaTeX formatting `r"$\sigma$ (S/m)"`
- Tab names: "σ vs P", "σ vs T", "σ vs m"

### 3. Auto-Generated Plots

**Removed:**
- "Create Bode and Nyquist Plots" button

**New Behavior:**
- Bode and Nyquist plots automatically generate when data rows are selected
- σ vs T and σ vs m plots also update automatically
- No manual button click required

The auto-generation is triggered by the table selection change signal:
```python
self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)
```

### 4. New Plot Types

#### σ vs T (Conductivity vs Temperature)
- Plots conductivity against temperature for all data points
- Color-coded by pressure if pressure data available
- Useful for analyzing temperature dependence of conductivity
- Located in the "σ vs T" tab

#### σ vs m (Conductivity vs Molality)
- Plots conductivity against total molality
- For multi-component solutions, uses sum of molalities
- Color-coded by temperature if temperature data available
- Useful for analyzing concentration dependence
- Located in the "σ vs m" tab

Both plots automatically set y-axis minimum to 0 (conductivity cannot be negative).

## Usage

### Viewing Data
1. Open the GUI (first tab shows data table by default)
2. Data table takes full window for easy viewing and editing
3. Click other tabs to view different plot types

### Generating Plots
1. Select one or more rows in the data table
2. Plots automatically generate in their respective tabs
3. Switch between tabs to view different visualizations

### Editing Data
All editing controls are in the Data Table tab:
- **Clear Selections** - Clear table row selection
- **Mark as Standard** - Mark selected rows as calibration standards
- **Associate Measurements** - Apply cell constant to measurements
- **Bulk Edit Comp/Conc** - Edit composition/concentration in bulk
- **Reload from CSV** - Reload data from config file
- **Export Plots to PDF** - Export all plots to PDF files

## Code Changes

### Key Functions Added

1. **`on_table_selection_changed()`**
   - Triggered when table selection changes
   - Auto-generates Bode/Nyquist plots
   - Updates σ vs T and σ vs m plots

2. **`refresh_sigma_vs_t_plot()`**
   - Generates conductivity vs temperature scatter plot
   - Colors points by pressure if available
   - Handles missing data gracefully

3. **`refresh_sigma_vs_m_plot()`**
   - Generates conductivity vs molality scatter plot
   - Sums molalities for multi-component solutions
   - Colors points by temperature if available

### Modified Structure

**Previous layout:**
```python
combined_layout = QHBoxLayout()
combined_layout.addLayout(self.table_layout)
combined_layout.addWidget(self.tabs)
```

**New layout:**
```python
# Table moved into its own tab
self.data_table_layout.addWidget(self.table)
# ... add buttons ...
self.data_table_tab.setLayout(self.data_table_layout)

# Main layout contains only tabs
main_layout.addWidget(self.tabs)
```

## Testing

A test suite has been created at `tests/test_gui_reorganization.py` with the following tests:

- `test_data_table_tab_exists()` - Verify Data Table is first tab
- `test_all_tabs_present()` - Verify all 6 tabs exist
- `test_sigma_symbol_in_dataframe()` - Verify σ symbol in columns
- `test_table_in_data_table_tab()` - Verify table is in correct tab
- `test_svt_figure_exists()` - Verify σ vs T plot exists
- `test_svm_figure_exists()` - Verify σ vs m plot exists
- `test_bode_nyquist_button_removed()` - Verify button was removed
- `test_auto_plot_on_selection()` - Verify auto-plot connection

Run tests with:
```bash
pytest tests/test_gui_reorganization.py -v
```

## LaTeX Table Changes

Updates to `generate_cortes_latex_tables.py`:

### Changes Made:
1. **Removed impedance columns** from Gamry data tables
   - Previous: `w, T, P, Z, σ` columns
   - New: `w, T, P, σ` columns (no Z column)

2. **Removed McCleskey comparison tables**
   - `generate_mccleskey_comparison_table()` no longer called
   - McCleskey model comparisons not included in output

3. **Added benchtop data support** (placeholder)
   - `generate_benchtop_table()` function added
   - Reads from `JesusData2025.csv`
   - Full parsing implementation needed (CSV format is complex)

### Files Generated:
- `cortes_nacl_table.tex` - NaCl impedance data (no Z column)
- `cortes_mgso4_table.tex` - MgSO4 impedance data (no Z column)
- `cortes_mixture_table.tex` - Mixture data
- `cortes_benchtop_table.tex` - Benchtop data (placeholder)

## Future Improvements

### High Priority:
1. **Complete benchtop data parsing**
   - JesusData2025.csv has complex sparse format
   - Needs custom parser for scattered data layout
   - Should verify against original source

2. **Add error bars to new plots**
   - σ vs T plot should show uncertainty bars
   - σ vs m plot should show uncertainty bars

### Medium Priority:
1. **Plot export functionality**
   - Add ability to export individual plot tabs
   - Include σ vs T and σ vs m in PDF exports

2. **Plot customization**
   - Allow user to toggle color coding
   - Add legend customization options

3. **Performance optimization**
   - Cache plot data to avoid regenerating on tab switches
   - Debounce selection changes to avoid excessive redraws

### Low Priority:
1. **Additional plot types**
   - σ vs P colored by molality
   - 3D surface plots (σ vs P vs T)
   - Arrhenius plots (log σ vs 1/T)

## Troubleshooting

### Plots not updating
- Ensure rows are selected in the data table
- Check that data has valid σ, T, P, or molality values
- Look for error messages in terminal/log

### Missing data in plots
- σ vs T requires valid T and σ values
- σ vs m requires valid molality and σ values
- NaN values are automatically filtered out

### Syntax warnings
All escape sequence warnings have been fixed by using raw strings (r"") for LaTeX formatting.

## References

- Original GUI: `hipoz_data_selector_gui.py`
- LaTeX tables: `generate_cortes_latex_tables.py`
- Test suite: `tests/test_gui_reorganization.py`
- Benchtop data: `JesusData2025.csv`
