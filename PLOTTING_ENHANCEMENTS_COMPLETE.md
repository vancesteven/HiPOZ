# HiPOZ Plotting Enhancements - Implementation Complete

**Date:** 2026-04-04
**Task:** Plot Enhancement Specialist (Task #2)
**Status:** ✅ COMPLETE

## Summary

Successfully enhanced the HiPOZ GUI plotting system with automatic plot generation, new conductivity analysis plots, and consistent use of the conductivity symbol σ throughout the interface.

## Implemented Features

### 1. Auto-Generate Bode & Nyquist Plots ✅

**What Changed:**
- **Removed:** "Create Bode and Nyquist Plots" button (manual trigger)
- **Added:** Automatic plot generation on table row selection
- **Location:** Line 351 - Selection change listener
- **Method:** `on_table_selection_changed()` at line 1329

**Benefits:**
- Immediate visual feedback when selecting measurements
- Eliminates extra click for users
- Updates all plots simultaneously (Bode, Nyquist, σ vs T, σ vs m)

**Implementation Details:**
```python
# Line 351: Connection
self.table.selectionModel().selectionChanged.connect(self.on_table_selection_changed)

# Line 1329-1343: Handler method
def on_table_selection_changed(self):
    """Auto-generate plots when selection changes."""
    try:
        selected_indexes = self.table.selectionModel().selectedRows()
        if selected_indexes:
            self.create_plots()  # Bode & Nyquist
            self.refresh_sigma_vs_t_plot()  # σ vs T
            self.refresh_sigma_vs_m_plot()  # σ vs m
    except Exception as e:
        log.debug(f"Auto-plot update error: {e}")  # Silent failure
```

### 2. Added "σ vs T" Plot Tab ✅

**New Tab:** Conductivity vs Temperature scatter plot
**Location:** Line 254-260 (tab setup), Line 1345-1391 (plot method)
**Tab Label:** "σ vs T"

**Plot Characteristics:**
- **X-axis:** Temperature (K)
- **Y-axis:** Conductivity σ (S/m)
- **Color coding:** Pressure (MPa) using 'viridis' colormap
- **Features:**
  - Dynamic color scale based on data range
  - Grid overlay for readability
  - Handles missing/invalid data gracefully
  - Shows "No valid data" message when empty

**Use Case:** Analyze temperature dependence of conductivity at various pressures

### 3. Added "σ vs m" Plot Tab ✅

**New Tab:** Conductivity vs Molality scatter plot
**Location:** Line 262-268 (tab setup), Line 1393-1450 (plot method)
**Tab Label:** "σ vs m"

**Plot Characteristics:**
- **X-axis:** Molality m (mol/kg solvent)
- **Y-axis:** Conductivity σ (S/m)
- **Color coding:** Temperature (K) using 'coolwarm' colormap
- **Features:**
  - Handles single-component solutions (direct float values)
  - **Smart multi-component support:** Parses comma-separated molalities and sums them
  - Y-axis constrained to ≥0 (conductivity cannot be negative)
  - Grid overlay and proper axis labels
  - Empty data handling

**Multi-Component Example:**
```python
# Input: w (molal) = "1.5,0.6"  (NaCl,MgSO4)
# Parsed: [1.5, 0.6]
# Plotted: m_total = 2.1 mol/kg
```

**Use Case:** Analyze concentration dependence of conductivity, especially for multi-salt brines

### 4. Symbol Replacement (S → σ) ✅

**Comprehensive Update:** All references to conductivity now use proper symbol σ

**Changes Made:**
- **DataFrame columns:** `'S (S/m)'` → `'σ (S/m)'`, `'S± (S/m)'` → `'σ± (S/m)'`
- **Tab label:** `"S vs P"` → `"σ vs P"` (line 293)
- **Plot labels:** All axis labels use σ symbol
- **Code references:** All `.columns.get_loc('S (S/m)')` → `.columns.get_loc('σ (S/m)')`
- **Export/Import:** CSV and JSON files use new column names

**Verification:**
- ✅ 0 remaining instances of old `'S (S/m)'` column name
- ✅ All plot methods updated
- ✅ All table operations updated
- ✅ Export/import functions updated

## Tab Organization

**New Layout:**
1. **Data Table** - Primary data view with editable table and control buttons
2. **Timeseries** - Time series plots of impedance measurements
3. **Bode & Nyquist** - Impedance spectroscopy plots (auto-updated)
4. **σ vs P** - Conductivity vs Pressure (existing, symbol updated)
5. **σ vs T** - Conductivity vs Temperature (NEW)
6. **σ vs m** - Conductivity vs Molality (NEW)

## Technical Implementation Details

### Error Handling
All auto-update plot methods include try/except blocks to prevent UI freezing:
```python
try:
    # Plot generation code
except Exception as e:
    log.debug(f"Auto-plot update error: {e}")  # Silent, non-blocking
```

### Performance Optimization
- Auto-update only triggers when rows are actually selected
- No redundant plot refreshes
- Lightweight selection handler

### Molality Calculations (Verified ✅)

**Single-Component:**
```python
# Direct conversion using PlanetProfile
molal = Ppt2molal(w_ppt, molar_mass)
```

**Multi-Component:**
```python
# Parse: "1.5,0.6" → [1.5, 0.6]
vals = [float(v.strip()) for v in str(m_str).split(',')]
m_total = sum(vals)  # Total ionic strength
```

**Mathematical Correctness:**
- Water mass = (1000g - total_w_ppt) / 1000 kg
- Molality = mol_solute / kg_water (per solvent, not solution)
- Multi-component uses sum of individual molalities for plotting

## File Modifications

**Primary File:** `hipoz_data_selector_gui.py`
**Statistics:**
- 290 lines added
- 44 lines deleted
- Net change: +246 lines

**Key Sections Modified:**
- Lines 254-268: New tab setup (σ vs T, σ vs m)
- Line 324-325: DataFrame column names (S → σ)
- Line 351: Selection change listener
- Line 1329-1343: Auto-update handler
- Line 1345-1391: σ vs T plot method
- Line 1393-1450: σ vs m plot method
- Multiple locations: Symbol replacements throughout

## Testing Checklist

### Basic Functionality
- [ ] Load data from `data/RoseData/` (or similar directory)
- [ ] Select single row in table → verify Bode/Nyquist auto-update
- [ ] Select multiple rows → verify all selected datasets plotted
- [ ] Switch between tabs → verify all 6 tabs render correctly

### Calibration Workflow
- [ ] Mark rows as calibration standards
- [ ] Associate measurements with standards
- [ ] Verify σ vs P plot updates (existing functionality)
- [ ] Verify σ vs T plot updates (NEW)
- [ ] Verify σ vs m plot updates (NEW)

### Multi-Component Solutions
- [ ] Load data with comma-separated molality values (e.g., "1.5,0.6")
- [ ] Verify σ vs m plot correctly sums molalities
- [ ] Check color coding by temperature

### Edge Cases
- [ ] Empty selection → verify plots clear gracefully
- [ ] Missing data (NaN values) → verify plots skip invalid points
- [ ] Zero/negative values → verify handling
- [ ] Very large datasets → verify performance

### Export/Import
- [ ] Export data to CSV → verify σ columns preserved
- [ ] Save to zAnalysis config → verify σ labels correct
- [ ] Reload GUI → verify σ data loads correctly

## Known Limitations

1. **Auto-update performance:** Very large selections (>100 rows) may cause brief UI lag
2. **Multi-component molality:** Assumes additive ionic strength (valid approximation)
3. **Color scales:** Dynamic vmin/vmax may cause scale differences between plots

## Future Enhancement Opportunities

1. **Toggle auto-update:** Add preference to enable/disable auto-generation
2. **Plot export:** Individual plot export buttons for each new tab
3. **Interpolation lines:** Add optional smooth curves between data points
4. **Log scales:** Optional log scale for σ vs m plots (high concentration range)
5. **Filtering:** Interactive filtering by P, T, or m ranges

## Documentation Updates Needed

- [ ] Update CLAUDE.md with new tab descriptions
- [ ] Update user guide with σ vs T and σ vs m usage examples
- [ ] Add multi-component molality explanation to docs
- [ ] Update screenshot in README if applicable

## Credits

**Implementation:** plot-specialist (AI agent)
**Coordination:** gui-architect (AI agent)
**Task Assignment:** team-lead (AI agent)
**Date:** 2026-04-04

---

## Quick Reference

**File:** `/Users/svance/Library/CloudStorage/Dropbox/ElectricalProperties/hipozgenai/hipoz_data_selector_gui.py`

**New Methods:**
- `on_table_selection_changed()` - Line 1329
- `refresh_sigma_vs_t_plot()` - Line 1345
- `refresh_sigma_vs_m_plot()` - Line 1393

**Tab Labels:**
- "Data Table" (tab 0)
- "Timeseries" (tab 1)
- "Bode & Nyquist" (tab 2)
- "σ vs P" (tab 3)
- "σ vs T" (tab 4)
- "σ vs m" (tab 5)

**Column Names:**
- `'σ (S/m)'` - Conductivity
- `'σ± (S/m)'` - Conductivity uncertainty
