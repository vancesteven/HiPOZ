# HiPOZ GUI Architecture

**Last Updated:** 2026-04-04

## Overview

The HiPOZ data selector GUI (`hipoz_data_selector_gui.py`) provides an interactive interface for calibrating conductivity measurements and curating impedance spectroscopy data.

## Current Architecture (As of 2026-04-04)

### Tab Structure

The GUI uses a horizontal split layout:

**Left Panel: Data Table**
- Interactive spreadsheet-style table
- 12 columns: Filename, Calibration, Time, Comp, w(ppt), w(molal), T(K), P(MPa), Z(Ohm), Z±(Ohm), S(S/m), S±(S/m)
- 7 control buttons below table

**Right Panel: Visualization Tabs**
1. **Timeseries Tab** - Resistance vs time plot
2. **Bode & Nyquist Tab** - Impedance spectroscopy plots
3. **S vs P Tab** - Conductivity vs pressure scatter plot

### Layout Code Structure

```python
# Current implementation (hipoz_data_selector_gui.py, lines 223-365)

# Create tab widget
self.tabs = QTabWidget()
self.timeseries_tab = QWidget()
self.plots_tab = QWidget()
self.svp_tab = QWidget()

# Add tabs
self.tabs.addTab(self.timeseries_tab, "Timeseries")
self.tabs.addTab(self.plots_tab, "Bode & Nyquist")
self.tabs.addTab(self.svp_tab, "S vs P")

# Create table layout
self.table_layout = QVBoxLayout()
self.table_layout.addWidget(self.table)
# Add 7 buttons...

# Combine horizontally
combined_layout = QHBoxLayout()
combined_layout.addLayout(self.table_layout)  # Left
combined_layout.addWidget(self.tabs)           # Right
```

### Data Table Columns

| Column | Type | Editable | Description |
|--------|------|----------|-------------|
| Filename | str | No | Gamry data file name |
| Calibration | str | Auto | Shows calibration group/role |
| Time | datetime | No | Measurement timestamp |
| Comp | str | Yes | Compound name(s), comma-separated for mixtures |
| w (ppt) | float/str | Yes | Parts per thousand (g/kg solution) |
| w (molal) | float/str | Yes | Molality (mol/kg solvent) |
| T (K) | float | Yes | Temperature in Kelvin |
| P (MPa) | int | Yes | Pressure in megapascals |
| Z (Ohm) | float | No | Fitted resistance |
| Z± (Ohm) | float | No | Resistance uncertainty (%) |
| S (S/m) | float | No | Conductivity (computed) |
| S± (S/m) | float | No | Conductivity uncertainty (%) |

**Key Features:**
- Auto-conversion between ppt ↔ molal for recognized compounds
- Multi-component solutions: comma-separated values (e.g., "NaCl,MgSO4")
- Empty cells display as blank (not "None" or "0")
- Zero values for P/T treated as missing data

### Control Buttons

1. **Clear Selections** - Deselect all table rows
2. **Mark as Standard** - Designate selected rows as calibration standards
3. **Associate Measurements** - Apply cell constant to selected measurements
4. **Bulk Edit Comp/Conc** - Edit composition for multiple rows
5. **Reload from CSV** - Reload config file
6. **Create Bode and Nyquist Plots** - Generate impedance plots for selected rows
7. **Export Plots to PDF** - Save plots to file

### Workflow

```
1. Load Data
   └─> Parse Gamry files
       └─> Populate table with P, T, Z values

2. Mark Standards
   └─> Select standard rows
       └─> Click "Mark as Standard"
           └─> Compute cell constant K_cell = σ_std × R

3. Associate Measurements
   └─> Select measurement rows
       └─> Click "Associate Measurements"
           └─> Compute conductivity σ = K_cell / R

4. Visualize
   └─> Switch to plot tabs
       └─> Review timeseries, Bode, Nyquist, S vs P
           └─> Export plots
```

## Proposed Architecture (Future Enhancement)

### New Tab Structure

Replace horizontal split with tab-based layout:

1. **Data Table** (NEW - first tab, default view)
   - Full-width table
   - All 7 control buttons
   - No space competition with plots

2. **Timeseries** (existing, second tab)
   - Resistance vs time
   - Error bars

3. **Bode & Nyquist** (existing, third tab)
   - Magnitude and phase plots
   - Complex plane plot

4. **S vs P** (existing, fourth tab)
   - Conductivity vs pressure
   - Temperature colormap

### Benefits

✅ **Full-width table** - No horizontal space constraints
✅ **Clear workflow** - Data entry → visualization
✅ **Consistent UX** - All content in tabs
✅ **Scalable** - Easy to add more tabs
✅ **Better ergonomics** - Dedicated space for each task

### Implementation Plan

See `/docs/GUI_REDESIGN_PROPOSAL.md` for detailed implementation steps.

## Data Table Implementation

### Auto-Conversion Logic

```python
# When user edits w(ppt) column
if comp and ',' in comp:
    # Multi-component
    molal_str = convert_multicomp_ppt_to_molal(comp, ppt_str)
else:
    # Single component (PlanetProfile integration)
    molal = Ppt2molal(comp, ppt_val)

# When user edits w(molal) column
if comp and ',' in comp:
    # Multi-component (returns total ppt)
    ppt_val = convert_multicomp_molal_to_ppt(comp, molal_str)
else:
    # Single component
    ppt = Molal2ppt(comp, molal_val)
```

### Multi-Component Solutions

For mixtures like "NaCl,MgSO4":
- **w(ppt)**: Single total value (e.g., 150 = 150 g total solute per kg solution)
- **w(molal)**: Comma-separated (e.g., "1.5,0.6" = 1.5 mol NaCl, 0.6 mol MgSO4 per kg water)

Conversion:
```
Total mass = 1000g water + Σ(molality_i × MW_i)
w_ppt_total = (Σ mass_i / Total mass) × 1000
```

### Calibration Groups

The "Calibration" column shows:
- Empty: No calibration role
- "std-GroupA": Calibration standard in group A
- "meas-GroupA": Measurement associated with group A

Groups allow bracketing: measure standards before and after samples, compute averaged cell constant.

## Plot Types

### 1. Timeseries Plot

**Location:** First tab
**X-axis:** Time (datetime)
**Y-axis (left):** Resistance (Ohm)
**Y-axis (right):** Conductivity (S/m, if available)
**Features:**
- Error bars (percent uncertainties)
- Dual y-axes for R and σ
- Interactive point selection (picker events)
- Highlights selected measurements

**Code:** `gamryPlots.plot_timeseries()`

### 2. Bode Plot

**Location:** Second tab (top panel)
**Panels:**
- Top: |Z| vs frequency (log-log)
- Bottom: Phase vs frequency (semilog)

**Features:**
- Full frequency sweep visualization
- Verifies circuit fit quality
- Identifies relaxation processes

### 3. Nyquist Plot

**Location:** Second tab (bottom panel)
**X-axis:** Re(Z) (Ohm)
**Y-axis:** -Im(Z) (Ohm)
**Features:**
- Equal aspect ratio (circle appears circular)
- Complex plane representation
- Fitted circuit overlay (optional)

### 4. S vs P Plot

**Location:** Third tab
**X-axis:** Pressure (MPa)
**Y-axis:** Conductivity (S/m)
**Color:** Temperature (K)
**Features:**
- Scatter plot with colormap
- Reveals pressure/temperature dependence
- Useful for phase diagram exploration

## Configuration Files

### Auto-Creation

When processing `data/<date>/`, GUI checks for:
1. `data/<date>/zAnalysis<date>.csv` (CSV format, Excel-friendly)
2. `data/<date>/zAnalysis<date>.json` (JSON format)

If neither exists, creates `zAnalysis<date>.csv` with empty template.

### Format Harmonization

- Load CSV → Save CSV
- Load JSON → Save JSON
- No cross-contamination

### Config Structure (CSV)

```csv
group_name,filename,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
GroupA,std_001.txt,standard,0.0084,KCl,1000,,,"84 µS/cm bottle"
GroupA,meas_001.txt,measurement,,NaCl,500,,,
```

### Config Structure (JSON)

```json
{
  "standards": [
    {
      "group": "GroupA",
      "filename": "std_001.txt",
      "conductivity_Sm": 0.0084,
      "comp": "KCl",
      "w_ppt": 1000
    }
  ],
  "measurements": [
    {
      "group": "GroupA",
      "filename": "meas_001.txt",
      "comp": "NaCl",
      "w_ppt": 500
    }
  ]
}
```

## State Management

### Masks

GUI maintains boolean arrays to track row states:

```python
self.standard_mask = np.zeros(n, dtype=bool)    # Row is a standard
self.associated_mask = np.zeros(n, dtype=bool)  # Row has associated σ
```

Updated when:
- User clicks "Mark as Standard"
- User clicks "Associate Measurements"
- Config file is loaded

### Auto-Save

GUI automatically saves state to config file:
- After marking standards
- After associating measurements
- After bulk edit operations
- On window close

Enables **reproducible analysis** by re-running with same config.

## Extension Points

### Adding New Plot Types

1. Create new QWidget for tab
2. Add FigureCanvas for matplotlib
3. Register tab with `self.tabs.addTab()`
4. Implement plot function in `gamryPlots.py`

Example:
```python
# In init_ui()
self.new_tab = QWidget()
self.new_layout = QVBoxLayout()
self.new_figure = Figure()
self.new_canvas = FigureCanvas(self.new_figure)
self.new_layout.addWidget(self.new_canvas)
self.new_tab.setLayout(self.new_layout)
self.tabs.addTab(self.new_tab, "New Plot")
```

### Adding New Table Columns

1. Add column to DataFrame initialization (line ~289)
2. Update `refresh_table()` display logic
3. Handle in `on_table_item_changed()` if editable
4. Update config save/load functions

### Custom Calibration Workflows

Override these methods:
- `mark_as_standard()` - Custom standard designation
- `associate_measurements()` - Custom σ computation
- `compute_cell_constant()` - Custom K_cell algorithm

## See Also

- [CALIBRATION.md](CALIBRATION.md) - Calibration workflow details
- [TESTING.md](../tests/README.md) - GUI test suite
- [MCP_USAGE_GUIDE.md](MCP_USAGE_GUIDE.md) - MCP integration
