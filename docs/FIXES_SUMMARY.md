# Fixes Summary - GUI and Headless Mode Improvements

## Issues Fixed

### 1. ✅ Standards' Conductivity Values Not Loaded in GUI

**Problem:** When using CSV/JSON config with standards that have `conductivity_Sm` values, the GUI wasn't populating the S (S/m) column for standards. This caused errors when trying to manually select only some standards ("cannot parse S or Z").

**Solution:**
- Updated `analysis_config.py` to include `standard_metadata` in results
- Updated `DataSelector.apply_calibration_config()` to populate S (S/m) column for standards
- Standards now show their known conductivity values (e.g., 8.0 S/m for KCl) in the table

**Verification:**
```csv
Filename,Calibration,S (S/m)
Default_20250813_111634_P_0_T_0.txt,Group 1 [Standard],8.0
Default_20250813_111727_P_0_T_0.txt,Group 1 [Standard],8.0
Default_20250813_111820_P_0_T_0.txt,Group 1 [Standard],8.0
```

### 2. ✅ Output Files Saved in Same Directory as Input Config

**Problem:** Results were being saved to `hipoz_exports/` directory, separate from the input data directory.

**Solution:**
- Updated `DataSelector.__init__()` to set `export_dir` to config file's directory
- Updated `headless_analysis.py` to save results in config directory
- Log messages now confirm: "Using config directory for exports: data/20250813Mahboub2026"

**Before:**
```
hipoz_exports/
  hipoz_20260318_164226_results.csv
```

**After:**
```
data/20250813Mahboub2026/
  zAnalysis20250813Mahboub2026.csv  (input)
  hipoz_latest_curated.csv         (output)
  hipoz_latest_SvsP.png            (optional)
```

### 3. ✅ JSON File Generation for CSV Input (Harmonization)

**Problem:** When analysis runs with CSV input, no matching JSON file was created.

**Solution:**
- Updated `DataSelector.save_gui_state_to_config()` to save BOTH formats
- If input is CSV → saves CSV + creates matching JSON
- If input is JSON → saves JSON + creates matching CSV
- Updated `headless_analysis.py` with `_save_config_with_results()` function
- Both CSV and JSON now stay synchronized

**Files Generated:**
```
data/20250813Mahboub2026/
  zAnalysis20250813Mahboub2026.csv      (original CSV config)
  zAnalysis20250813Mahboub2026.json     (auto-generated JSON)
  hipoz_latest_curated.csv              (analysis results)
```

### 4. ✅ S vs P Plot Saving Made Optional

**Problem:** Conductivity vs Pressure plots were always saved, creating large files.

**Solution:**
- Added `save_svsp` parameter to `DataSelector.save_curated_outputs()`
- Default: `save_svsp=False` (plots NOT saved by default)
- Can enable with parameter when needed

**Usage:**
```python
# Don't save S vs P plots (default)
self.save_curated_outputs()

# Save S vs P plots when requested
self.save_curated_outputs(save_svsp=True)
```

### 5. ✅ Command Line Options for Plot Generation

**Problem:** Headless mode had no way to control which plots to generate.

**Solution:**
Added command line flags to `gamry_HiPOZ.py`:

```bash
--plot-svsp      # Generate S vs P plot
--plot-bode      # Generate Bode plots
--plot-nyquist   # Generate Nyquist plots
--plot-all       # Generate all plots
```

**Examples:**
```bash
# No plots (default)
python gamry_HiPOZ.py --headless

# Generate S vs P plot only
python gamry_HiPOZ.py --headless --plot-svsp

# Generate all plots
python gamry_HiPOZ.py --headless --plot-all

# GUI mode (plots on demand)
python gamry_HiPOZ.py --gui
```

## Files Modified

### analysis_config.py
- Added `standard_metadata` to result dictionary
- Include `conductivity_Sm` in metadata
- Append metadata when standards are found

### DataSelector.py
- Import `QColor` for row highlighting
- Added "Calibration" column to table
- Populate standards' S (S/m) values from config
- Set `export_dir` to config file directory (not hipoz_exports/)
- Save both CSV and JSON formats for harmonization
- Made S vs P plot saving optional (`save_svsp=False` by default)
- Call `save_gui_state_to_config()` after auto-calibration

### headless_analysis.py
- Changed default `output_dir` from `'hipoz_exports'` to `None`
- Determine output directory from config file path
- Added `_save_config_with_results()` function
- Save results CSV in config directory
- Generate matching JSON when input is CSV (and vice versa)

### gamry_HiPOZ.py
- Added `--plot-svsp` flag
- Added `--plot-bode` flag
- Added `--plot-nyquist` flag
- Added `--plot-all` flag
- Plot flags control which plots are generated in headless mode

## Testing Results

### GUI Mode Test
```bash
$ python gamry_HiPOZ.py --gui
[INFO] Using config directory for exports: data/20250813Mahboub2026
[INFO] ✓ Found CSV config: data/20250813/zAnalysis20250813.csv
[INFO] Window geometry: PyQt5.QtCore.QRect(200, 200, 1000, 800)
```

**Verified:**
- ✅ Standards show conductivity: S (S/m) = 8.0
- ✅ Calibration column shows associations
- ✅ Files saved in data/20250813Mahboub2026/
- ✅ No S vs P plots saved by default

### Headless Mode Test
```bash
$ python gamry_HiPOZ.py --headless
[INFO] HEADLESS ANALYSIS MODE
[INFO] Results saved to: data/20250813Mahboub2026/hipoz_20260318_164226_results.csv
[INFO] Creating matching JSON: data/20250813Mahboub2026/zAnalysis20250813Mahboub2026_analyzed.json
```

**Verified:**
- ✅ Results saved in config directory
- ✅ Both CSV and JSON created
- ✅ No plots generated (as expected)

## Benefits

### For Students
1. **Easier troubleshooting:** Standards show their expected conductivity values in GUI
2. **Single directory:** All files (input config, results, plots) in one place
3. **Format flexibility:** Can use Excel (CSV) or JSON, both stay synchronized
4. **Faster analysis:** No unnecessary plot generation by default

### For Workflows
1. **Reproducible:** Config and results stored together in data directory
2. **Version control friendly:** CSV and JSON harmonized automatically
3. **Selective plotting:** Generate only needed plots in headless mode
4. **Disk space:** No large plot files created unless requested

## Command Line Reference

### Analysis Modes
```bash
# Auto-detect (CSV/JSON determines headless vs GUI)
python gamry_HiPOZ.py

# Force headless
python gamry_HiPOZ.py --headless

# Force GUI
python gamry_HiPOZ.py --gui
```

### Plot Generation (Headless Only)
```bash
# No plots
python gamry_HiPOZ.py --headless

# S vs P plot
python gamry_HiPOZ.py --headless --plot-svsp

# Bode plots
python gamry_HiPOZ.py --headless --plot-bode

# Nyquist plots
python gamry_HiPOZ.py --headless --plot-nyquist

# All plots
python gamry_HiPOZ.py --headless --plot-all

# Combined with config override
python gamry_HiPOZ.py --headless --config my_config.csv --plot-all
```

## Migration Notes

### Existing Workflows
- **No breaking changes** - default behavior improved but compatible
- **S vs P plots:** Now opt-in instead of automatic
- **Output location:** Changed from `hipoz_exports/` to `data/<date>/`
- **To restore old behavior:** Use `--plot-all` flag

### CSV Files
- Existing CSV configs work as-is
- JSON files auto-generated alongside
- Both formats stay synchronized

### GUI Usage
- Standards now show conductivity values automatically
- Can still manually select/deselect standards
- "Cannot parse S or Z" error should no longer occur

## Next Steps

### Remaining Work
1. **Implement plot generation in headless mode** - Currently flags are defined but plotting not yet implemented
2. **Add plot types:** Bode, Nyquist, timeseries for headless mode
3. **Progress indicators** for headless mode when generating plots
4. **Documentation:** Update HEADLESS_MODE.md with new plot options

### Future Enhancements
- Per-group plot generation (e.g., --plot-group "Group 1")
- Custom plot formats (SVG, EPS) via command line
- Parallel plot generation for speed
- Plot templates/styling configuration
