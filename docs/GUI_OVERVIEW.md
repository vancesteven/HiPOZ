# HiPOZ Data Selector GUI Overview

## Introduction

The HiPOZ Data Selector GUI (`hipoz_data_selector_gui.py`) provides an interactive interface for impedance data analysis, calibration, and visualization. This document describes the GUI structure, features, and usage patterns.

## GUI Architecture

### Tab Structure

The GUI organizes functionality into four main tabs:

1. **Data Table** (Default view)
   - Full-window spreadsheet view of all measurements
   - Editable columns for composition, concentration, P, T
   - Calibration group assignments
   - Action buttons for workflow operations

2. **Timeseries**
   - Impedance vs time plot
   - Color-coded by composition
   - Error bars showing measurement uncertainty
   - Interactive selection for detailed analysis

3. **Bode & Nyquist**
   - Bode plots (|Z| and phase vs frequency)
   - Nyquist plots (Re(Z) vs -Im(Z))
   - Circuit fit overlays
   - Generated for selected table rows

4. **S vs P**
   - Conductivity vs pressure scatter plot
   - Color-coded by temperature
   - Shows calibrated measurements only

### Layout Design

**Current Structure (Side-by-side):**
```
┌─────────────────────────────────────┐
│  Table + Buttons  │  Timeseries     │
│                   │  Bode & Nyquist │
│                   │  S vs P         │
└─────────────────────────────────────┘
```

**Proposed Structure (Tabbed):**
```
┌─────────────────────────────────────┐
│ [Data Table] [Timeseries] [Bode] [S vs P] │
├─────────────────────────────────────┤
│                                     │
│     Full-width content area         │
│                                     │
└─────────────────────────────────────┘
```

The proposed tabbed structure gives the data table full window width, eliminating horizontal space competition with plots.

## Data Table Features

### Column Structure

| Column | Description | Editable | Format |
|--------|-------------|----------|--------|
| **Filename** | Original data file | No | Text (basename only) |
| **Calibration** | Group and role (std/meas) | Automatic | Text |
| **Time** | Measurement timestamp | No | ISO datetime |
| **Comp** | Composition | Yes | Text (e.g., 'NaCl', 'KCl') |
| **w (ppt)** | Concentration (g/kg) | Yes | Float with precision matching |
| **w (molal)** | Molality (mol/kg H₂O) | Yes | Float with precision matching |
| **T (K)** | Temperature | Yes | Float (shows blank if 0) |
| **P (MPa)** | Pressure | Yes | Integer (shows blank if 0) |
| **Z (Ohm)** | Fitted impedance | No | Float |
| **Z± (Ohm)** | Impedance uncertainty | No | Percentage |
| **S (S/m)** | Conductivity | No | Float (blank until calibrated) |
| **S± (S/m)** | Conductivity uncertainty | No | Percentage |

### Special Display Rules

- **Empty cells**: Display as blank (not "None" or "0")
- **Zero P/T**: Show as empty (zero doesn't make physical sense)
- **Precision matching**: w(ppt) precision matches w(molal) precision (min 2 decimal places)
- **Multi-component**: Support comma-separated values (e.g., "NaCl,KCl" with "1.5,0.6" molal)

### Interactive Features

**Row Selection:**
- Click row to select
- Shift-click to select range
- Ctrl-click to select multiple non-contiguous rows
- Selection persists across tabs

**Cell Editing:**
- Double-click to edit
- Tab to move to next cell
- Changes auto-save to config file
- Invalid entries show warning message

**Right-click Context Menu:**
- Copy selected cells
- Paste values
- Clear selection
- Mark as standard/measurement
- Exclude from analysis

## Workflow Operations

### 1. Mark as Standard

**Purpose:** Designate calibration measurements with known conductivity.

**Steps:**
1. Select rows containing KCl standard measurements
2. Click "Mark as Standard" button
3. Dialog prompts for:
   - Calibration group name
   - Known conductivity (S/m)
   - Composition (default: KCl)
   - Concentration

**Result:**
- Calculates cell constant: K_cell = σ_known × R_measured
- Updates "Calibration" column: "GroupName (std)"
- Saves to config file for reproducibility

**Common Standards:**

| Label | Conductivity (S/m) | µS/cm |
|-------|-------------------|-------|
| KCl 23 | 0.0023 | 23 |
| KCl 84 | 0.0084 | 84 |
| KCl 447 | 0.0447 | 447 |
| KCl 2070 | 0.2070 | 2070 |

Conversion: µS/cm ÷ 10000 = S/m

### 2. Associate Measurements

**Purpose:** Calculate conductivity for measurements using calibration group cell constant.

**Steps:**
1. Ensure at least one calibration group has standards
2. Select measurement rows to calibrate
3. Click "Associate Measurements" button
4. Dialog prompts for:
   - Which calibration group to use
   - Confirm composition/concentration

**Result:**
- Calculates conductivity: σ = K_cell / R_measured
- Updates "S (S/m)" and "S± (S/m)" columns
- Updates "Calibration" column: "GroupName (meas)"
- Saves to config file

**Uncertainty Propagation:**
- Combines impedance fit uncertainty
- Includes cell constant standard deviation
- Displays as percentage in "S±" column

### 3. Bulk Edit Comp/Conc

**Purpose:** Quickly set composition/concentration for multiple rows.

**Steps:**
1. Select rows with same composition
2. Click "Bulk Edit Comp/Conc" button
3. Dialog prompts for:
   - Composition (NaCl, KCl, MgSO4, etc.)
   - Concentration in molal or ppt
   - Multi-component: comma-separated values

**Result:**
- Updates "Comp", "w (molal)", and "w (ppt)" columns
- Automatic unit conversion
- Precision matching for multi-component solutions

**Multi-component Example:**
```
Comp: NaCl,MgSO4
w (molal): 1.5,0.6
→ w (ppt): 85.23,69.45  (precision matched)
```

### 4. Create Bode and Nyquist Plots

**Purpose:** Visualize impedance spectra and circuit fits for selected measurements.

**Steps:**
1. Select one or more table rows
2. Click "Create Bode and Nyquist Plots" button
3. View plots in "Bode & Nyquist" tab

**Bode Plot Features:**
- |Z| vs frequency (log-log scale)
- Phase angle vs frequency
- Circuit fit overlay (dashed line)
- Color-coded by row

**Nyquist Plot Features:**
- -Im(Z) vs Re(Z)
- Circuit fit overlay
- High-frequency region detail
- Electrode polarization visible at low frequency

### 5. Export Plots to PDF

**Purpose:** Save all plots for publication or reporting.

**Steps:**
1. Generate desired plots (Timeseries, Bode & Nyquist, S vs P)
2. Click "Export Plots to PDF" button
3. Choose export mode:
   - **Overwrite**: Always same filename
   - **Timestamp**: Unique filename per export
   - **Rolling**: Keep last N versions

**Output Location:**
- Config directory if available: `data/<date>/`
- Otherwise: `hipoz_exports/`

**Files Created:**
- `timeseries.pdf` - Time series plot
- `bode_nyquist.pdf` - Impedance plots
- `sigma_vs_pressure.pdf` - Conductivity plot
- `curated_data.csv` - Table export with all results

### 6. Reload from CSV

**Purpose:** Refresh GUI from saved config file (useful after manual edits).

**Steps:**
1. Edit `zAnalysis<date>.csv` in Excel or text editor
2. Save file
3. Click "Reload from CSV" button
4. GUI updates to match file contents

**Use Cases:**
- Correct typos in Excel
- Add/remove excluded files
- Adjust calibration groups
- Restore from backup

## Configuration Files

### zAnalysis<date>.csv Format

**Recommended format** for Excel users. One row per file.

```csv
group_name,filename,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,KCl_84.txt,standard,0.0084,KCl,1.0,,,"84 µS/cm standard"
Group 1,sample_1000.txt,measurement,,NaCl,1000.0,1.5,,"NaCl measurement"
Group 1,bad_data.txt,measurement,,,,,x,"Skip this file"
```

**Column Descriptions:**
- `group_name`: Calibration group (e.g., "Morning", "Group 1")
- `filename`: Data file name (basename, not full path)
- `type`: "standard" or "measurement"
- `conductivity_Sm`: Known σ for standards (S/m)
- `comp`: Composition string (e.g., "NaCl", "NaCl,KCl")
- `w_ppt`: Concentration (g/kg solution)
- `w_molal`: Molality (mol/kg H₂O)
- `exclude`: Mark with 'x', 'yes', 'true', or '1' to skip
- `notes`: Free text annotations

### zAnalysis<date>.json Format

Alternative format for programmatic access.

```json
{
  "description": "Analysis for 20250813",
  "date": "20250813",
  "calibrations": [
    {
      "name": "Group 1",
      "standards": [
        {
          "filename": "KCl_84.txt",
          "conductivity_Sm": 0.0084,
          "comp": "KCl",
          "w_ppt": 1.0
        }
      ],
      "measurements": [
        {
          "filename": "sample_1000.txt",
          "comp": "NaCl",
          "w_molal": 1.5
        },
        {
          "filename": "bad_data.txt",
          "exclude": true
        }
      ]
    }
  ]
}
```

### Auto-Creation and Persistence

**GUI Behavior:**
1. **On startup**: Check for `zAnalysis<date>.csv` or `.json` in `data/<date>/`
2. **If missing**: Create empty CSV template
3. **If found**: Load and apply calibration
4. **During work**: Save changes back to config file
5. **On exit**: Ensure final state persisted

**Format Priority:**
- CSV checked first (Excel-friendly)
- JSON as fallback
- Format preserved (CSV → CSV, JSON → JSON)
- No cross-contamination

### File Organization

```
data/
  20250813/
    ConductivityData_Default/
      Default_20250813_104603_P_0_T_0.txt    # KCl standard
      Default_20250813_124016_P_126_T_297.txt # NaCl measurement
      ...
    zAnalysis20250813.csv                    # Analysis config (CSV)
    # OR
    zAnalysis20250813.json                   # Analysis config (JSON)
    timeseries.pdf                           # Exported plot
    bode_nyquist.pdf                         # Exported plot
    sigma_vs_pressure.pdf                    # Exported plot
    curated_data.csv                         # Table export
```

## Status Bar

Bottom of window shows real-time feedback:

- **"Ready"** - Idle, waiting for user action
- **"Loading config file..."** - Reading zAnalysis file
- **"Applying calibration..."** - Calculating cell constants
- **"Marked N rows as standard"** - Operation complete
- **"Associated N measurements"** - Calibration applied
- **"Saved to <path>"** - Export successful
- **Error messages** - Red text for problems

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+S | Save current state to config file |
| Ctrl+R | Reload from config file |
| Ctrl+C | Copy selected cells |
| Ctrl+V | Paste into selected cells |
| Ctrl+A | Select all rows |
| Delete | Clear selected cell values |
| F5 | Refresh plots |
| Esc | Clear selection |

## Common Workflows

### Daily Analysis Routine

1. **Launch GUI**: `python gamry_HiPOZ.py`
2. **View data**: Check "Data Table" tab
3. **Mark standards**: Select KCl rows → "Mark as Standard"
4. **Associate measurements**: Select sample rows → "Associate Measurements"
5. **Review plots**: Check "S vs P" tab for outliers
6. **Export**: "Export Plots to PDF"

### Bracketed Measurements

When standards measured multiple times during experiment:

1. **Morning group**: Mark first KCl standard as "Group 1"
2. **Morning samples**: Associate with "Group 1"
3. **Afternoon group**: Mark second KCl standard as "Group 2"
4. **Afternoon samples**: Associate with "Group 2"
5. Each group gets independent cell constant

### Multi-component Solutions

For mixed electrolytes (e.g., ocean analog):

1. **Edit table**: Enter "NaCl,MgSO4" in Comp column
2. **Enter concentrations**: "1.5,0.6" in w(molal) column
3. **Auto-convert**: GUI calculates total ppt
4. **Precision matching**: ppt decimals match molal decimals

### Handling Bad Data

Files with problems (incomplete runs, electrical noise, instrument errors):

**Option 1: Exclude in config file**
```csv
group_name,filename,type,exclude,notes
Group 1,bad_data.txt,measurement,x,"Noisy data"
```

**Option 2: Don't associate**
- Mark standards as usual
- Skip bad measurement rows during "Associate Measurements"
- They remain uncalibrated (blank S values)

## Troubleshooting

### Table Not Updating

**Problem:** Edits don't save or reload overwrites changes

**Solutions:**
- Click outside cell to commit edit (press Enter)
- Check status bar for save confirmation
- Verify config file has write permissions
- Look for error messages in console

### Calibration Not Applied

**Problem:** "Associate Measurements" doesn't fill S values

**Solutions:**
- Verify at least one standard marked in same group
- Check standard has valid conductivity_Sm value
- Ensure impedance fit succeeded (check Z column)
- Review console log for calculation errors

### Plots Not Generating

**Problem:** "Create Plots" button does nothing

**Solutions:**
- Select at least one row before clicking
- Check selected rows have valid impedance data
- Verify matplotlib backend is working
- Look for Python exceptions in console

### Precision Mismatch

**Problem:** w(ppt) shows wrong number of decimals

**Solutions:**
- Enter w(molal) first (sets precision)
- Use minimum 2 decimals in molal
- For multi-component, all values need same precision
- Delete and re-enter with correct decimals

### Multi-component Issues

**Problem:** Comma-separated values not parsing

**Solutions:**
- No spaces after commas: "1.5,0.6" not "1.5, 0.6"
- Same number of values in Comp and concentration
- Check console for conversion errors
- Verify MOLAR_MASSES has all compounds

## Advanced Features

### Custom Export Modes

Edit `hipoz_data_selector_gui.py` line ~379:

```python
self.save_mode = "overwrite"  # or "timestamp" or "rolling"
self.rolling_keep = 5         # for rolling mode
```

### Adding New Compounds

Edit MOLAR_MASSES dictionary (line ~23):

```python
MOLAR_MASSES = {
    'NaCl': 58.44,
    'KCl': 74.55,
    'MgSO4': 120.37,
    'CaCl2': 110.98,  # Add new compound
}
```

### Customizing Table Columns

Modify `self.data` DataFrame creation (line ~289):

```python
self.data = pd.DataFrame({
    'Filename': filenames_display,
    'CustomColumn': [None] * len(filenames_display),  # Add column
    # ... existing columns
})
```

## See Also

- [CALIBRATION.md](CALIBRATION.md) - Detailed calibration workflow
- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [FORMAT_HARMONIZATION.md](FORMAT_HARMONIZATION.md) - CSV/JSON synchronization
- [CLAUDE.md](../CLAUDE.md) - Full project documentation
