# HiPOZ Headless Analysis Mode

## Overview

HiPOZ now supports **headless mode** for automated analysis without GUI interaction. When a complete CSV/JSON configuration file exists with standards and measurements pre-specified, the analysis runs automatically and saves results.

## Usage

### Auto-Detection (Recommended)
```bash
python gamry_HiPOZ.py
```

**Behavior:**
- If config file has complete calibration setup → Runs headless
- If config missing or incomplete → Launches GUI

### Force Headless Mode
```bash
python gamry_HiPOZ.py --headless
```

**Requirements:**
- Config file must exist
- Must have at least one standard with `conductivity_Sm` value
- Must have at least one measurement

**Error:** Exits with error if config is incomplete

### Force GUI Mode
```bash
python gamry_HiPOZ.py --gui
```

**Use Cases:**
- Manual review of auto-calibrated data
- Making adjustments to configurations
- Visual inspection of impedance spectra
- Interactive data curation

## Workflow

### 1. Set Up Configuration (One Time)

Create CSV config in your data directory:

```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,KCl_std_1.txt,0,293,standard,8.0,KCl,,,,Calibration standard
Group 1,KCl_std_2.txt,0,293,standard,8.0,KCl,,,,Calibration standard
Group 1,sample_1.txt,10,293,measurement,,NaCl,,1.0,,1 molal NaCl
Group 1,sample_2.txt,20,293,measurement,,NaCl,,1.0,,1 molal NaCl
Group 1,sample_3.txt,30,293,measurement,,NaCl,,1.0,,1 molal NaCl
```

**File Location:** `data/<date>/zAnalysis<date>.csv`

Example: `data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv`

### 2. Run Analysis

```bash
python gamry_HiPOZ.py
```

**Processing Steps:**
1. Loads all impedance data files
2. Fits CPE circuit models
3. Detects complete config → enters headless mode
4. Computes cell constant from standards: `K = σ_std × R`
5. Applies to measurements: `σ = K / R`
6. Saves results to `hipoz_exports/hipoz_TIMESTAMP_results.csv`

### 3. Results

**Output File:** `hipoz_exports/hipoz_20260318_164226_results.csv`

**Columns:**
- `group_name`: Calibration group identifier
- `filename`: Data file name
- `P_MPa`: Pressure in megapascals
- `T_K`: Temperature in Kelvin
- `type`: "standard" or "measurement"
- `comp`: Composition (e.g., "KCl", "NaCl")
- `w_ppt`, `w_molal`: Concentration (ppt or molal)
- `R_ohm`, `R_ohm_unc`: Measured resistance and uncertainty
- `K_cell`, `K_cell_unc`: Cell constant and uncertainty
- `conductivity_Sm`, `conductivity_Sm_unc`: Computed conductivity and uncertainty
- `exclude`: Exclusion flag
- `notes`: Additional notes

## GUI Pre-Population

When launching GUI with an existing config:

```bash
python gamry_HiPOZ.py --gui
```

**Visual Indicators:**
- **Light blue rows**: Pre-configured standards
- **Light green rows**: Pre-configured measurements
- **Tooltips**: Hover over first column to see calibration group

**Auto-Applied:**
- Cell constant computation from standards
- Conductivity calculation for measurements
- Composition and concentration metadata

**Benefits:**
- See which files are already configured
- Verify calibration associations
- Make adjustments if needed
- Visual confirmation of setup

## Configuration File Formats

### CSV Format (Recommended for Excel)

```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,file1.txt,0,293,standard,8,KCl,,,,
Group 1,file2.txt,10,293,measurement,,NaCl,,1.0,,
```

**Advantages:**
- Easy editing in Excel/Google Sheets
- Visual layout
- No syntax errors

### JSON Format (For Programmatic Use)

```json
{
  "calibration_groups": [
    {
      "name": "Group 1",
      "standards": [
        {
          "filename": "file1.txt",
          "P_MPa": 0,
          "T_K": 293,
          "conductivity_Sm": 8.0,
          "comp": "KCl"
        }
      ],
      "measurements": [
        {
          "filename": "file2.txt",
          "P_MPa": 10,
          "T_K": 293,
          "comp": "NaCl",
          "w_molal": 1.0
        }
      ]
    }
  ]
}
```

**Advantages:**
- Structured data
- Programmatic generation
- Complex configurations

## Excluding Files

Mark files to exclude from analysis:

**CSV:**
```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,bad_data.txt,0,0,measurement,,NaCl,,1.0,x,Equipment malfunction
```

**JSON:**
```json
{
  "filename": "bad_data.txt",
  "exclude": true,
  "notes": "Equipment malfunction"
}
```

**Effect:** File is loaded and fit but excluded from calibration and analysis.

## Multiple Calibration Groups

For time-series data with drift correction:

```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,std_morning_1.txt,0,293,standard,8,KCl,,,,Morning calibration
Group 1,std_morning_2.txt,0,293,standard,8,KCl,,,,
Group 1,sample_1.txt,10,293,measurement,,NaCl,,1.0,,
Group 1,sample_2.txt,20,293,measurement,,NaCl,,1.0,,
Group 2,std_afternoon_1.txt,0,293,standard,8,KCl,,,,Afternoon recalibration
Group 2,std_afternoon_2.txt,0,293,standard,8,KCl,,,,
Group 2,sample_3.txt,30,293,measurement,,NaCl,,1.0,,
Group 2,sample_4.txt,40,293,measurement,,NaCl,,1.0,,
```

**Each group:**
- Computes independent cell constant
- Applies to associated measurements only
- Compensates for drift between calibration periods

## Example Session

```bash
$ python gamry_HiPOZ.py
[INFO] Found CSV analysis config: data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv
[INFO] Loading calibration configuration from: data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv
[INFO] Loaded 1 calibration group(s) from CSV
[INFO]   Group 1: 3 standards, 6 measurements
[INFO] Processing 20250813Mahboub2026/
[INFO] Found 9 data files to process
Processing measurement 1 of 9: Default_20250813_120000_P_0_T_0.txt
...
[INFO] Successfully processed 9 of 9 files for 20250813Mahboub2026
[INFO] TimeSeries data organized successfully

======================================================================
HEADLESS ANALYSIS MODE
Config file has standards and measurements - running without GUI
======================================================================

[INFO] Running headless analysis mode
[INFO] Found 9 solution objects

Processing Group 1
[INFO]   Standards: 3
[INFO]   Measurements: 6
[INFO]   KCl_std_1.txt: σ=8.00e+00 S/m, R=1.15e+01 Ω → K=9.22e+01 S/m·Ω
[INFO]   KCl_std_2.txt: σ=8.00e+00 S/m, R=1.15e+01 Ω → K=9.22e+01 S/m·Ω
[INFO]   KCl_std_3.txt: σ=8.00e+00 S/m, R=1.15e+01 Ω → K=9.22e+01 S/m·Ω
[INFO]   Cell constant: K = 9.22e+01 ± 3.87e-02 S/m·Ω
[INFO]   (from 3 standard(s))
[INFO]   sample_1.txt: R=1.11e+01 Ω → σ=8.28e+00 ± 4.48e-03 S/m
[INFO]   sample_2.txt: R=1.96e+01 Ω → σ=4.70e+00 ± 4.02e-03 S/m
...

Results saved to: hipoz_exports/hipoz_20260318_164226_results.csv
[INFO] Total measurements analyzed: 6
[INFO] Total standards: 3

======================================================================
ANALYSIS COMPLETE
======================================================================
Results have been saved. Use --gui flag to visualize or make adjustments.
```

## Benefits

### For Students
- ✅ Set up analysis once in Excel
- ✅ Run analysis with single command
- ✅ Reproducible workflow
- ✅ No GUI interaction needed
- ✅ Easy to batch process multiple dates

### For Automation
- ✅ Scriptable analysis pipeline
- ✅ Consistent results
- ✅ Version-controlled configurations
- ✅ Integration with data processing workflows

### For Research
- ✅ Documented analysis parameters
- ✅ Traceable calibration
- ✅ Uncertainty propagation
- ✅ Quality control via review mode (--gui)

## Troubleshooting

### "Headless analysis failed: no valid standards"

**Cause:** Standards missing `conductivity_Sm` value

**Fix:**
```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,std_1.txt,0,0,standard,8,KCl,,,,← Add conductivity value
```

### "Config file not found"

**Cause:** No `zAnalysis<date>.csv` or `.json` in data directory

**Fix:** Create config file in `data/<date>/zAnalysis<date>.csv`

### "Cannot run headless: no measurements"

**Cause:** Config only has standards, no measurements marked

**Fix:** Add measurement rows with `type=measurement`

### P and T values are 0

**Cause:** Filenames don't include P and T information

**Fix:** Either:
1. Use standard naming: `sample_P_10_T_293.txt`
2. Manually add P_MPa and T_K values to CSV

## See Also

- `FORMAT_HARMONIZATION.md` - CSV/JSON format details
- `HARMONIZATION_SUMMARY.md` - Configuration management
- `analysis_config.py` - Configuration file API
- `headless_analysis.py` - Headless mode implementation
