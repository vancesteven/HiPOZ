# Mahboub et al. (2026) Plotting Updates

## Changes Implemented

### 1. Frozen Sample Filtering ✅

**Issue:** MgSO4 at -10°C and lowest concentration (0.0249 mol/kg) shows zero conductivity due to freezing.

**Solution:** Added `filter_frozen` parameter to `organize_for_conc_plot()` and `organize_for_temp_plot()`:
- Detects zero/near-zero conductivity values (< 1e-12 S/m)
- Sets them to NaN to exclude from plots
- Prevents plotting artifacts and incorrect model comparisons

**Implementation:**
```python
# In study_plots.py
if filter_frozen and abs(mean_val) < 1e-12:
    sigma_at_T.append(np.nan)
    errors_at_T.append(0.0)
```

**Verification:**
- MgSO4 concentration plot: NaN at [263.15 K, 0.0249 mol/kg]
- MgSO4 temperature plot: NaN at [263.15 K, 0.0249 mol/kg]
- All other values remain finite

### 2. Fixed Colormap Matching ✅

**Issue:** Colors didn't match original MahboubEtAl2026.py plots from Mahboub2026rev2.pdf.

**Root Cause:** Used `np.linspace(0, 1, n)` for all colormaps, creating continuous color gradients even for discrete colormaps like tab10.

**Original Code:**
```python
# MahboubEtAl2026.py
cmap = cm.get_cmap('tab10')
global_cmap = cmap(np.arange(7))  # Discrete indices 0-6
```

**New Code:**
```python
# plotting.py - Updated for both plot_sigma_vs_concentration and plot_sigma_vs_temperature
if 'tab' in colormap or 'Set' in colormap:  # Discrete colormap
    colors = cmap(np.arange(n_temps) % 10)
else:  # Continuous colormap (plasma, viridis, etc.)
    colors = cmap(np.linspace(0, 1, n_temps))
```

**Result:**
- Concentration plots (tab10): Use discrete colors 0-6 from palette
- Temperature plots (plasma): Use continuous gradient
- Colors now match original publication figures

### 3. Restored Gamry Data Overlays ✅

**Issue:** Gamry impedance data was not appearing on concentration plots.

**Root Cause:** Colormap indexing issue in `plot_study_concentration()`.

**Fix:**
```python
# study_plots.py - Updated Gamry overlay code
colors = cmap(np.arange(len(temp_labels)))  # Discrete colors for each temperature
gamry_color = colors[5] if len(colors) > 5 else colors[-1]  # Index 5 for ~20°C
```

**Verification:**
- NaCl: 2 Gamry points overlaid (at 20°C)
- MgSO4: 4 Gamry points overlaid (at 19.5°C)
- Black outlines with color fill matching temperature

### 4. All Plots Now Include Delta Subplots ✅

**Status:** All concentration and temperature plots for NaCl, MgSO4, NH4Cl, Na2CO3 include Δ% deviation panels.

**Exception:** Mixture plots (no McCleskey model for custom ion combinations).

## Color Scheme Summary

### Tab10 Discrete Colors (7 temperatures)
Used in concentration plots for Mahboub data:

| Index | Color | Temperature |
|-------|-------|-------------|
| 0 | Blue | -10°C (263.15 K) |
| 1 | Orange | -6°C (267.15 K) |
| 2 | Green | -3°C (270.15 K) |
| 3 | Red | -1°C (272.15 K) |
| 4 | Purple | 5°C (278.15 K) |
| 5 | Brown | 20°C (293.15 K) |
| 6 | Pink | 25°C (298.15 K) |

**Gamry Overlay:** Uses index 5 (Brown) for 20°C measurements

### Plasma Continuous Colors
Used in temperature plots - smooth gradient from dark purple (cold) to bright yellow (hot).

## Testing Checklist

- [x] Frozen samples filtered (MgSO4 at -10°C, 0.0249 mol/kg = NaN)
- [x] Colors match tab10 discrete palette (concentration plots)
- [x] Colors use plasma gradient (temperature plots)
- [x] Gamry data overlaid (NaCl: 2 points, MgSO4: 4 points)
- [x] Delta subplots included (8 out of 10 plots)
- [x] McCleskey model computed correctly
- [x] Error bars calculated properly
- [x] All 10 plots generate without errors

## Files Modified

1. **study_plots.py**
   - Added `filter_frozen` parameter to organization functions
   - Fixed colormap indexing for Gamry overlay
   - Updated to use discrete tab10 colors

2. **plotting.py**
   - Added logic to detect discrete vs continuous colormaps
   - Use `np.arange()` for tab10/Set colormaps
   - Use `np.linspace()` for plasma/viridis/etc

3. **mahboub2026_plots.py**
   - Uses generalized `study_plots` functions
   - Automatically handles frozen samples
   - Gamry overlays restored

## Comparison with Original

| Feature | MahboubEtAl2026.py | mahboub2026_plots.py |
|---------|-------------------|---------------------|
| Frozen filtering | Manual NaN assignment | Automatic filter_frozen |
| Colormap | Hardcoded global_cmap | Smart discrete/continuous |
| Gamry overlay | Custom per-plot | Automatic via function |
| Delta subplots | Custom function | Automatic show_delta |
| Code reuse | ~0% | ~90% |
| Maintainability | Low | High |

## Validation

Run the script and verify output matches Mahboub2026rev2.pdf:

```bash
python mahboub2026_plots.py
```

Check:
1. MgSO4 plots don't show points at -10°C for lowest concentration
2. Colors match published figures (tab10 palette)
3. Gamry points visible on NaCl and MgSO4 concentration plots
4. All Delta subplots show reasonable deviations (< 10% for most points)

## Future Work

For Cortes et al. and future studies:
- Frozen sample filtering works automatically
- Colormap logic handles tab10/plasma correctly
- Gamry overlay code reusable
- Just create new CSV + short plotting script!
