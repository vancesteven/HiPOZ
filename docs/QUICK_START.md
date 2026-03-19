# HiPOZ Quick Start Guide

## Daily Analysis Workflow

Each day's measurements get their own analysis file: `zAnalysis<date>.csv` or `zAnalysis<date>.json`

**CSV format is recommended for Excel users!**

### Option A: Let the GUI Auto-Create (Recommended for New Data)

```bash
python gamry_HiPOZOZ.py
```

The GUI will automatically:
- ✅ Create `zAnalysis<date>.csv` in `data/<date>/` if it doesn't exist
- ✅ Format ready for Excel editing
- ✅ Save your calibration work as you mark standards and associate measurements
- ✅ Persist GUI state for reproducible analysis

### Option B: Pre-Generate Config File (For Custom Setup)

**Generate CSV (Excel-friendly, default):**
```bash
python calibration_config.py scan data/20250813
```

This creates: `zAnalysis20250813.csv` - Open in Excel!

**Or generate JSON:**
```bash
python calibration_config.py scan data/20250813 --format json
```

Then move to data directory:
```bash
mv zAnalysis20250813.csv data/20250813/
```

### Step 2: Edit Analysis File (Optional)

**For CSV files (Excel):**

Open `data/20250813/zAnalysis20250813.csv` in Excel:

| group_name | filename | type | conductivity_Sm | comp | w_ppt | w_molal | exclude | notes |
|------------|----------|------|-----------------|------|-------|---------|---------|-------|
| Group 1 | KCl_84.txt | standard | 0.0084 | KCl | 1.0 | | | 84 µS/cm |
| Group 1 | NaCl_sample.txt | measurement | | NaCl | 1000 | | | |
| Group 1 | bad_run.txt | measurement | | | | | x | Skip this file |

**Key columns:**
- **type**: `standard` or `measurement`
- **conductivity_Sm**: For standards (e.g., 0.0084 for 84 µS/cm)
- **comp**: Composition (e.g., KCl, NaCl, MgSO4)
- **w_ppt**: Concentration in g/kg (parts per thousand)
- **exclude**: Mark with `x` to skip files from analysis
- **group_name**: Create multiple groups for bracketed measurements

**For JSON files:**

Open `data/20250813/zAnalysis20250813.json` in text editor:

```json
{
  "description": "Analysis for 20250813",
  "date": "20250813",
  "calibrations": [
    {
      "name": "Morning measurements",
      "standards": [
        {
          "filename": "Default_20250813_104603_P_0_T_0.txt",
          "conductivity_Sm": 0.0084,
          "comp": "KCl",
          "w_ppt": 1.0
        }
      ],
      "measurements": [
        {
          "filename": "Default_20250813_124016_P_126_T_297.txt",
          "comp": "NaCl",
          "w_ppt": 1000
        },
        {
          "filename": "bad_run.txt",
          "exclude": true
        }
      ]
    }
  ]
}
```

### Step 3: Run Analysis

```bash
python gamry_HiPOZOZ.py
```

The system automatically:
- ✅ Detects or creates `zAnalysis20250813.json` in data/20250813/
- ✅ Applies calibration from existing config (if standards specified)
- ✅ Fills GUI table with results
- ✅ Generates S vs P plot
- ✅ Saves to hipoz_exports/
- ✅ Saves GUI progress back to zAnalysis file as you work

## File Organization

```
data/
  20250813/
    ConductivityData_Default/
      *.txt                           # Your measurement files
    zAnalysis20250813.csv             # Analysis config (Excel-friendly, preferred)
    # OR zAnalysis20250813.json       # Analysis config (JSON format)
  20250814/
    ConductivityData_Default/
      *.txt
    zAnalysis20250814.csv             # Analysis config for next day
```

## Why zAnalysis<date>?

- **z prefix**: Sorts to bottom of directory listings
- **Date suffix**: Clear which day's data this analyzes
- **One per day**: Standards from same day are used for that day's measurements
- **Comprehensive**: Contains all analysis parameters in one place
- **Auto-created**: GUI creates empty template if missing and saves your work
- **Persistent**: Re-running with same config reproduces the analysis
- **CSV format**: Excel-friendly for easy editing by students
- **Exclude column**: Mark files to skip (bad data, test runs, etc.)

## Workflow Notes

**First-time analysis:**
1. Run `python gamry_HiPOZOZ.py` - GUI auto-creates `zAnalysis<date>.json`
2. Use GUI to mark standards and associate measurements
3. GUI saves your work back to the config file
4. Result: Reproducible analysis stored in config file

**Subsequent runs:**
1. Run `python gamry_HiPOZOZ.py` - GUI loads existing `zAnalysis<date>.json`
2. Calibration automatically applied from saved config
3. Make adjustments in GUI if needed
4. GUI saves updates back to config file

## Common KCl Standards

| Label | Conductivity (S/m) | JSON Value |
|-------|--------------------| ---------- |
| 84 µS/cm | 0.0084 | `"conductivity_Sm": 0.0084` |
| 447 µS/cm | 0.0447 | `"conductivity_Sm": 0.0447` |
| 2070 µS/cm | 0.2070 | `"conductivity_Sm": 0.2070` |

**Conversion:** µS/cm ÷ 10000 = S/m

## Concentration Units

- **w_ppt**: parts per thousand (g solute / kg solution)
  - Example: `"w_ppt": 1000` = 1000 g/kg

- **w_molal**: molal (mol solute / kg solvent)
  - Example: `"w_molal": 1.5` = 1.5 mol/kg

## Excluding Files from Analysis

Sometimes you want to skip certain files (bad data, test runs, incomplete measurements):

**In CSV (Excel):**
- Mark the `exclude` column with `x`, `yes`, `true`, or `1`
- Files marked for exclusion will be skipped during analysis

Example:
```csv
group_name,filename,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,bad_data.txt,measurement,,,,,x,Bad data - skip
Group 1,test_run.txt,measurement,,,,,yes,Test run only
```

**In JSON:**
- Add `"exclude": true` to any file entry
- Works for both standards and measurements

Example:
```json
"measurements": [
  {"filename": "good_data.txt", "comp": "NaCl", "w_ppt": 1000},
  {"filename": "bad_data.txt", "exclude": true}
]
```

## Multiple Calibration Groups (Bracketing)

If you measured standards multiple times during the day:

```json
{
  "calibrations": [
    {
      "name": "Before lunch",
      "standards": [
        {"filename": "KCl_84_morning.txt", "conductivity_Sm": 0.0084}
      ],
      "measurements": [
        {"filename": "sample_1000.txt", "comp": "NaCl", "w_ppt": 1000}
      ]
    },
    {
      "name": "After lunch",
      "standards": [
        {"filename": "KCl_84_afternoon.txt", "conductivity_Sm": 0.0084}
      ],
      "measurements": [
        {"filename": "sample_1100.txt", "comp": "NaCl", "w_ppt": 1100}
      ]
    }
  ]
}
```

Each group gets its own cell constant!

## Command Reference

```bash
# Generate CSV analysis file (Excel-friendly, default)
python calibration_config.py scan data/20250813

# Generate JSON analysis file
python calibration_config.py scan data/20250813 --format json

# Run analysis (auto-detects CSV or JSON zAnalysis files)
python gamry_HiPOZOZ.py

# Specify config explicitly (CSV or JSON)
python gamry_HiPOZOZ.py --config data/20250813/zAnalysis20250813.csv
python gamry_HiPOZOZ.py --config data/20250813/zAnalysis20250813.json

# Generate example files (for reference)
python calibration_config.py
```

## Format Comparison

| Feature | CSV | JSON |
|---------|-----|------|
| **Excel editing** | ✅ Native | ❌ Need text editor |
| **Student-friendly** | ✅ Familiar | ❌ Less familiar |
| **Quick edits** | ✅ Spreadsheet | ❌ Text file |
| **Exclude files** | ✅ Mark with 'x' | ✅ Set true |
| **Multiple groups** | ✅ Easy rows | ❌ Nested arrays |
| **Auto-created by GUI** | ✅ Default | ✅ Optional |

**Recommendation:** Use CSV for ease of editing in Excel

## Troubleshooting

**"No calibration config found"**
- Check file is named `zAnalysis<date>.csv` or `zAnalysis<date>.json`
- Check file is in `data/<date>/` directory
- GUI will auto-create if missing - just run `python gamry_HiPOZOZ.py`
- Use `--config` flag to specify path explicitly

**"No standards found"**
- Edit your zAnalysis file (CSV in Excel, JSON in text editor)
- Mark KCl files as `type=standard` (CSV) or move to "standards" array (JSON)
- Add conductivity_Sm values (e.g., 0.0084 for 84 µS/cm)

**"No valid cell constants"**
- Check conductivity values are correct (84 µS/cm = 0.0084 S/m)
- Verify standard files loaded successfully
- Check impedance fits didn't fail (look at log output)

**"File still analyzed even though I marked it to exclude"**
- CSV: Make sure exclude column has 'x', 'yes', 'true', or '1'
- JSON: Make sure `"exclude": true` (lowercase, boolean not string)
- Save the file after editing!

**"Can't edit CSV in Excel"**
- Make sure file has `.csv` extension
- If it opens as text, use "Open With..." → Excel
- Or right-click → Edit With → Excel

## Format Harmonization

**Important:** The system preserves your format choice!

- ✅ Load CSV → GUI saves to CSV
- ✅ Load JSON → GUI saves to JSON
- ✅ No cross-contamination between formats
- ✅ Both stored in same directory: `data/<date>/`

**When both exist:** CSV has priority (Excel-friendly)

See `FORMAT_HARMONIZATION.md` for details on how CSV, JSON, and GUI stay synchronized.

## See Also

- `FORMAT_HARMONIZATION.md` - How CSV/JSON/GUI harmonize
- `CALIBRATION_WORKFLOW.md` - Detailed calibration guide
- `CALIBRATION_README.md` - Complete documentation
- `CLAUDE.md` - Full project documentation
