# Cortes Plot Fixes - April 29, 2026

## Issues Fixed

### 1. Mixture Compound Title Formatting
**Problem:** Titles for mixtures like `Na2SO4:KCl_2:1` were missing subscripts and had underscores instead of spaces.

**Solution:** Added proper LaTeX formatting in `cortes2026_plots.py`:
- `Na2SO4:KCl_2:1` → `Na$_2$SO$_4$:KCl 2:1` (subscripts applied, underscore replaced with space)
- `Na2SO4:KCl_1:1` → `Na$_2$SO$_4$:KCl 1:1`
- `Na2SO4:KCl_1:2` → `Na$_2$SO$_4$:KCl 1:2`
- `NaCl+Glycine` → `NaCl+Glycine` (no changes needed)

**Result:** Titles now display correctly with subscripts and proper spacing.

### 2. McCleskey Predictions for Mixtures
**Problem:** Salt mixtures were not showing McCleskey model comparisons.

**Solution:** 
- Created `get_mixture_ion_spec()` function to generate ion specifications for mixtures based on their molar ratios
- For `Na2SO4:KCl 2:1`, the ion spec calculates:
  - Na⁺: 2 × (2/3) = 4/3 (from 2 parts Na₂SO₄)
  - SO₄²⁻: 2/3 (from 2 parts Na₂SO₄)
  - K⁺: 1/3 (from 1 part KCl)
  - Cl⁻: 1/3 (from 1 part KCl)
- Updated `study_plots.py` to accept custom `ion_spec` parameter
- Enabled show_delta for salt mixtures (but not for organic mixtures like NaCl+Glycine)

**Result:** Salt mixtures now display McCleskey comparison lines and Δ% deviation subplots. Glycine mixtures correctly exclude McCleskey (glycine not in model).

### 3. EIS Data Overlay with Pressure Filtering
**Problem:** 1 bar EIS (electrochemical impedance spectroscopy) data was not being overlaid on benchtop plots.

**Solution:**
- Added configuration options in `cortes2026_plots.py`:
  ```python
  INCLUDE_EIS_OVERLAY = True  # Toggle EIS overlay on/off
  EIS_PRESSURE_MAX = 1.0      # Filter to ≤1 MPa (≈10 bar)
  ```
- Updated `extract_compound_overlay_data()` to:
  - Accept `pressure_max` parameter for filtering
  - Convert concentration data from strings to floats (fixed dtype issue)
  - Remove NaN values before plotting
- Added informative output messages showing how many EIS points were added

**Result:** Low-pressure EIS measurements (≤1 MPa) are now overlaid on benchtop concentration plots. Users can disable overlay by setting `INCLUDE_EIS_OVERLAY = False`.

## Files Modified

1. **cortes2026/cortes2026_plots.py**
   - Added configuration flags for EIS overlay
   - Added `get_mixture_ion_spec()` function
   - Updated `extract_compound_overlay_data()` with pressure filtering and type conversion
   - Updated compound LaTeX formatting for mixtures
   - Added ion_spec support for mixture plotting

2. **study_plots.py**
   - Added `ion_spec` parameter to `plot_study_concentration()` and `plot_study_temperature()`
   - Updated McCleskey model computation to use custom ion specs for mixtures
   - Modified model computation calls to pass ion_spec when provided

## Testing

All plots regenerated successfully:
- ✅ Mixture titles display with correct subscripts and spacing
- ✅ McCleskey comparisons shown for salt mixtures
- ✅ NaCl+Glycine correctly excludes McCleskey
- ✅ EIS overlay adds ~3 points to NaCl concentration plot
- ✅ No errors during plot generation

## Configuration Options

Users can control EIS overlay behavior by editing `cortes2026_plots.py`:

```python
# Disable EIS overlay entirely
INCLUDE_EIS_OVERLAY = False

# Adjust pressure filter (e.g., only atmospheric pressure)
EIS_PRESSURE_MAX = 0.15  # MPa (≈1.5 bar)
```

### 4. EIS Data in McCleskey Deviation Subplot
**Problem:** EIS overlay data appeared in the main plot but not in the Δ% (McCleskey deviation) subplot.

**Solution:**
- Modified `plot_study_concentration()` in `study_plots.py` to compute McCleskey predictions for EIS data
- EIS measurements are assumed to be at 293.15 K (20°C, typical room temperature for 1 bar measurements)
- Δ% values are calculated and plotted in the bottom panel with the same styling as main plot
- Error handling added in case McCleskey model computation fails

**Result:** EIS data points now appear in both the main conductivity plot (top) and the McCleskey deviation plot (bottom).

**Example for NaCl:**
- EIS data at 0.5, 0.75, 2.0 mol/kg
- Δ% values: +7.2%, +8.8%, -7.3%
- Both panels show black circles with colored fill

## Example Output

For Na₂SO₄:KCl 2:1:
- **Before:** `Na2SO4:KCl_2:1 Conductivity vs Temperature` (no subscripts, underscore)
- **After:** `Na₂SO₄:KCl 2:1 Conductivity vs Temperature` (subscripts, space)

For NaCl with EIS:
- Console output: `EIS overlay: 3 points at P ≤ 1.0 MPa`
- Main plot (top): Black-edged circles for EIS data overlaid on benchtop measurements
- Delta plot (bottom): Same EIS points showing % deviation from McCleskey model
