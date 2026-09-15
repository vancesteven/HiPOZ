# Cortes Combined Plots - Improvements

**Date**: April 2, 2026  
**Status**: ✅ COMPLETE

## Changes Made

### 1. Y-Axis Minimum Set to Zero

**Files modified**:
- `plotting.py` - Updated two functions:
  - `plot_sigma_vs_concentration()` - Line ~333
  - `plot_sigma_vs_temperature()` - Line ~465

**Implementation**:
```python
# Set y-axis minimum to 0 (conductivity cannot be negative)
ax1.set_ylim(bottom=0)
```

**Rationale**: Conductivity is a physical quantity that cannot be negative. Setting y-axis minimum to 0 provides:
- Correct physical interpretation
- Better visual scale
- Consistent with McCleskey comparison plots
- Publication-ready appearance

### 2. Excluded Specific Amino Acids from Plots

**File modified**: `cortes2026/cortes2026_plots.py`

**Excluded compounds**:
- Alanine
- Glutamic Acid
- Aspartic Acid

**Retained compounds**:
- Glycine (kept in plots)
- NaCl+Glycine (kept in plots)

**Implementation**:
```python
# Exclude certain amino acids from plotting (but keep in tables)
exclude_from_plots = ['Alanine', 'Glutamic Acid', 'Aspartic Acid']

if benchtop_data:
    all_compounds = list(benchtop_data.keys())
    compounds_to_plot = [c for c in all_compounds if c not in exclude_from_plots]
```

**Rationale**: 
- Pure amino acids (Alanine, Glutamic Acid, Aspartic Acid) have very low conductivity
- Not relevant for main publication focus
- Data still available in CSV for tables/supplementary material
- Glycine retained as it's part of NaCl+Glycine comparison

### 3. Cleaned Up Output Directory

**Action**: Removed old amino acid plot files
```bash
rm cortes2026/cortes_plots/alanine_*.pdf
rm cortes2026/cortes_plots/aspartic_acid_*.pdf
rm cortes2026/cortes_plots/glutamic_acid_*.pdf
```

**Result**: Directory now contains only relevant plots

## Updated Plot Count

**Before**: 26 plots (13 compounds × 2 plot types)
**After**: 20 plots (10 compounds × 2 plot types)

### Plots Generated

**Single salts** (4 compounds):
- Glycine (2 plots)
- KCl (2 plots)
- NaCl (2 plots) ✓ with Gamry overlay
- Na2SO4 (2 plots)

**Mixtures** (5 compounds):
- Na2SO4:KCl_1:1 (2 plots)
- Na2SO4:KCl_2:1 (2 plots)
- NaCl:MgSO4_1:1 (2 plots)
- NaCl:MgSO4_1:2 (2 plots)
- NaCl:MgSO4_2:1 (2 plots)

**Organics** (1 compound):
- NaCl+Glycine (2 plots)

**Total**: 10 compounds × 2 plot types = **20 plots**

## File List

```
cortes2026/cortes_plots/
├── glycine_vs_concentration.pdf
├── glycine_vs_temperature.pdf
├── kcl_vs_concentration.pdf
├── kcl_vs_temperature.pdf
├── na2so4_vs_concentration.pdf
├── na2so4_vs_temperature.pdf
├── na2so4:kcl_1:1_vs_concentration.pdf
├── na2so4:kcl_1:1_vs_temperature.pdf
├── na2so4:kcl_2:1_vs_concentration.pdf
├── na2so4:kcl_2:1_vs_temperature.pdf
├── nacl_vs_concentration.pdf          ← with Gamry overlay
├── nacl_vs_temperature.pdf            ← with Gamry overlay
├── nacl+glycine_vs_concentration.pdf
├── nacl+glycine_vs_temperature.pdf
├── nacl:mgso4_1:1_vs_concentration.pdf
├── nacl:mgso4_1:1_vs_temperature.pdf
├── nacl:mgso4_1:2_vs_concentration.pdf
├── nacl:mgso4_1:2_vs_temperature.pdf
├── nacl:mgso4_2:1_vs_concentration.pdf
└── nacl:mgso4_2:1_vs_temperature.pdf
```

## Console Output

```
Detected compounds: Alanine, Aspartic Acid, Glutamic Acid, Glycine, KCl, 
                    Na2SO4, Na2SO4:KCl_1:1, Na2SO4:KCl_2:1, NaCl, 
                    NaCl+Glycine, NaCl:MgSO4_1:1, NaCl:MgSO4_1:2, NaCl:MgSO4_2:1
Plotting: Glycine, KCl, Na2SO4, Na2SO4:KCl_1:1, Na2SO4:KCl_2:1, NaCl, 
          NaCl+Glycine, NaCl:MgSO4_1:1, NaCl:MgSO4_1:2, NaCl:MgSO4_2:1
Excluded from plots: Alanine, Aspartic Acid, Glutamic Acid
```

## Data Preservation

**Important**: Excluded amino acids are still present in:
- ✅ `Cortes2026BenchtopData.csv` - All 132 rows preserved
- ✅ Available for LaTeX table generation
- ✅ Available for supplementary material
- ✅ Can be plotted on demand by removing from exclude list

No data was deleted, only plots were removed from the output directory.

## Benefits

1. **Cleaner plot set**: Focus on publication-relevant compounds
2. **Physical correctness**: Y-axis minimum enforced at zero
3. **Consistency**: Matches McCleskey comparison plot style
4. **Flexibility**: Easy to re-include amino acids if needed
5. **Data integrity**: Original data preserved in CSV

## Usage

Generate updated plots:
```bash
cd /work/hipozgenai/cortes2026
python3 cortes2026_plots.py
```

To include excluded amino acids:
```python
# In cortes2026_plots.py, modify or comment out:
exclude_from_plots = []  # Empty list includes all compounds
```

## Summary

✅ Y-axis minimum set to 0 for all conductivity plots  
✅ Alanine, Glutamic Acid, Aspartic Acid excluded from plots  
✅ Glycine and NaCl+Glycine retained  
✅ Old amino acid plot files removed  
✅ Plot count: 20 PDFs (10 compounds × 2 plot types)  
✅ Data preserved in CSV for tables  

**Status**: Complete and ready for publication
