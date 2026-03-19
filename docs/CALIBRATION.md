# Automated Calibration Configuration

HiPOZ supports automated calibration via configuration files with **explicit filename specification**, **multiple calibration groups** for bracketed measurements, and **file exclusion** for skipping bad data.

## Quick Start

### 1. Generate Example Config Files

```bash
python calibration_config.py
```

This creates:
- `calibration_config_example.csv` - **CSV format (recommended for Excel users)**
- `calibration_config_example.json` - JSON format

### 2. Auto-Generate Config from Your Data

**Generate CSV (Excel-friendly, default):**
```bash
python calibration_config.py scan data/20250813
```

**Or generate JSON:**
```bash
python calibration_config.py scan data/20250813 --format json
```

This scans your data directory and generates `zAnalysis<date>.csv` (or `.json`) with all your files listed. You then edit it to:
- Add conductivity values to standards
- Add composition and concentration to measurements
- Split into multiple calibration groups (for bracketing)
- Mark files to exclude with 'x' (bad data, test runs, etc.)
- Verify standards are correctly identified

### 3. Place Config in Data Directory

```
data/
  20250813/
    ConductivityData_Default/
      *.txt                          # Your measurement files
    zAnalysis20250813.csv            # Your config file (CSV recommended)
    # OR zAnalysis20250813.json      # JSON format also supported
```

**Note:** The GUI will auto-create this file if missing when you run `python gamry_HiPOZOZ.py`

### 4. Run HiPOZ

```bash
python gamry_HiPOZOZ.py
```

The GUI automatically:
- Detects CSV or JSON config (CSV preferred)
- Auto-creates config if missing
- Applies all calibration groups
- Skips excluded files
- Fills in the table with results
- Shows filenames in the leftmost column
- Generates S vs P plot
- Saves to `hipoz_exports/`
- Saves your GUI work back to the config file

## Config Format: Explicit Filenames

### CSV Format (Recommended for Excel)

**Open in Excel for easy editing!**

```csv
group_name,filename,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Morning,Default_20250813_0800_KCl_84uScm.txt,standard,0.0084,KCl,1.0,,,84 µS/cm
Morning,Default_20250813_0805_KCl_447uScm.txt,standard,0.0447,KCl,5.0,,,447 µS/cm
Morning,Default_20250813_0900_NaCl_1000ppt_P100.txt,measurement,,NaCl,1000,,,
Morning,Default_20250813_0930_NaCl_1000ppt_P150.txt,measurement,,NaCl,1000,,,
Morning,Default_20250813_0945_bad_data.txt,measurement,,,,,x,Bad run - exclude
Afternoon,Default_20250813_1400_KCl_84uScm.txt,standard,0.0084,KCl,1.0,,,
Afternoon,Default_20250813_1500_NaCl_2000ppt_P100.txt,measurement,,NaCl,2000,,,
```

**Key columns:**
- `group_name`: Which calibration group (for bracketing)
- `filename`: Exact filename (just basename, not full path)
- `type`: Either `standard` or `measurement`
- `conductivity_Sm`: For standards only (e.g., 0.0084 for 84 µS/cm)
- `comp`: Composition (KCl, NaCl, MgSO4, etc.)
- `w_ppt`: Concentration in g/kg (parts per thousand)
- `w_molal`: Concentration in mol/kg (alternative to w_ppt)
- **`exclude`**: Mark with `x`, `yes`, `true`, or `1` to skip this file
- `notes`: Free-form notes for your reference

### JSON Format (Alternative)

```json
{
  "description": "My experiment",
  "calibrations": [
    {
      "name": "Morning session",
      "standards": [
        {"filename": "Default_20250813_0800_KCl_84uScm.txt", "conductivity_Sm": 0.0084, "comp": "KCl", "w_ppt": 1.0},
        {"filename": "Default_20250813_0805_KCl_447uScm.txt", "conductivity_Sm": 0.0447, "comp": "KCl", "w_ppt": 5.0}
      ],
      "measurements": [
        {"filename": "Default_20250813_0900_NaCl_1000ppt_P100.txt", "comp": "NaCl", "w_ppt": 1000},
        {"filename": "Default_20250813_0930_NaCl_1000ppt_P150.txt", "comp": "NaCl", "w_ppt": 1000},
        {"filename": "Default_20250813_0945_bad_data.txt", "exclude": true}
      ]
    },
    {
      "name": "Afternoon session",
      "standards": [
        {"filename": "Default_20250813_1400_KCl_84uScm.txt", "conductivity_Sm": 0.0084, "comp": "KCl", "w_ppt": 1.0}
      ],
      "measurements": [
        {"filename": "Default_20250813_1500_NaCl_2000ppt_P100.txt", "comp": "NaCl", "w_ppt": 2000}
      ]
    }
  ]
}
```

## Multiple Calibration Groups (Bracketing)

**Why multiple groups?**

When running long experiments, you want to bracket measurements with standards measured before and after:

```
0800: Measure KCl standards (Group 1 standards)
0900-1100: Measure NaCl samples (Group 1 measurements)
1200: Measure KCl standards again (Group 2 standards)
1300-1500: Measure more NaCl samples (Group 2 measurements)
```

Each group gets its own cell constant calculated from its standards, accounting for drift.

### Example Config with Bracketing

```json
{
  "calibrations": [
    {
      "name": "Morning - low pressure series",
      "standards": [
        "KCl_84_before.txt",
        "KCl_447_before.txt"
      ],
      "measurements": [
        "NaCl_P100_T298.txt",
        "NaCl_P150_T298.txt",
        "NaCl_P200_T298.txt"
      ]
    },
    {
      "name": "Afternoon - high pressure series",
      "standards": [
        "KCl_84_after.txt",
        "KCl_447_after.txt"
      ],
      "measurements": [
        "NaCl_P250_T298.txt",
        "NaCl_P300_T298.txt"
      ]
    }
  ]
}
```

## GUI Features

### Filename Column

The GUI now shows filenames in the leftmost column, making it easy to:
- Identify which files are being analyzed
- Cross-reference with your config file
- Verify the correct files are selected

### Auto-Applied Calibration

When a config is detected:
1. ✅ All calibration groups are applied automatically
2. ✅ Cell constants calculated for each group
3. ✅ Conductivities computed for measurements
4. ✅ Results filled into the table
5. ✅ Summary dialog shows what was applied
6. ✅ S vs P plot generated and saved

### Summary Dialog

After auto-calibration, you'll see:

```
Applied 2 calibration group(s):

✓ Morning session: K_cell = 45.23 ± 0.15 1/m from 3 standard(s)
✓ Afternoon session: K_cell = 45.87 ± 0.18 1/m from 2 standard(s)

Total: 5 standards, 8 measurements
Results saved to hipoz_exports/
```

## Workflow Examples

### Example 1: Simple Single-Group Experiment

You measured standards at the start, then did all your measurements:

**1. Scan your data (generates CSV by default):**
```bash
python calibration_config.py scan data/20250813
```

**2. Edit `zAnalysis20250813.csv` in Excel:**
- Keep the single group
- Verify KCl files have `type=standard`
- Add conductivity_Sm values to standards (e.g., 0.0084 for 84 µS/cm)
- Verify sample files have `type=measurement`
- Add comp and w_ppt to measurements
- Mark any bad runs with `x` in exclude column

**3. Move to data directory:**
```bash
mv zAnalysis20250813.csv data/20250813/
```

**4. Run:**
```bash
python gamry_HiPOZOZ.py
```

### Example 2: Bracketed Measurements

You bracketed measurements with standards:

**Config:**
```json
{
  "calibrations": [
    {
      "name": "Pre-experiment standards + measurements",
      "standards": ["KCl_84_0800.txt", "KCl_447_0800.txt"],
      "measurements": ["NaCl_0900.txt", "NaCl_0930.txt"]
    },
    {
      "name": "Post-experiment standards + measurements",
      "standards": ["KCl_84_1300.txt", "KCl_447_1300.txt"],
      "measurements": ["NaCl_1400.txt", "NaCl_1430.txt"]
    }
  ]
}
```

Each group's measurements use only that group's standards, accounting for instrument drift.

### Example 3: Multiple Compositions with Bracketing

You measured different compositions with standards in between:

```json
{
  "calibrations": [
    {
      "name": "NaCl series",
      "standards": ["KCl_before_NaCl.txt"],
      "measurements": ["NaCl_1000ppt.txt", "NaCl_1500ppt.txt"]
    },
    {
      "name": "MgSO4 series",
      "standards": ["KCl_before_MgSO4.txt"],
      "measurements": ["MgSO4_500ppt.txt", "MgSO4_1000ppt.txt"]
    }
  ]
}
```

### Example 4: Time-Series with Regular Calibration Checks

```json
{
  "calibrations": [
    {
      "name": "Hour 0-2",
      "standards": ["KCl_0hr.txt"],
      "measurements": ["sample_0hr.txt", "sample_1hr.txt", "sample_2hr.txt"]
    },
    {
      "name": "Hour 2-4",
      "standards": ["KCl_2hr.txt"],
      "measurements": ["sample_2hr.txt", "sample_3hr.txt", "sample_4hr.txt"]
    },
    {
      "name": "Hour 4-6",
      "standards": ["KCl_4hr.txt"],
      "measurements": ["sample_4hr.txt", "sample_5hr.txt", "sample_6hr.txt"]
    }
  ]
}
```

### Example 5: Excluding Bad Data (CSV)

In Excel, mark files to skip:

| group_name | filename | type | conductivity_Sm | comp | w_ppt | exclude | notes |
|------------|----------|------|-----------------|------|-------|---------|-------|
| Group 1 | KCl_std.txt | standard | 0.0084 | KCl | 1.0 | | Good |
| Group 1 | sample_001.txt | measurement | | NaCl | 1000 | | Good |
| Group 1 | sample_002.txt | measurement | | NaCl | 1000 | x | Power outage - exclude |
| Group 1 | sample_003_test.txt | measurement | | | | yes | Test run only |
| Group 1 | sample_004.txt | measurement | | NaCl | 1000 | | Good |

Files with `x`, `yes`, `true`, or `1` in exclude column will be skipped.

### Example 6: Excluding Bad Data (JSON)

```json
{
  "calibrations": [
    {
      "name": "Group 1",
      "standards": [
        {"filename": "KCl_std.txt", "conductivity_Sm": 0.0084}
      ],
      "measurements": [
        {"filename": "sample_001.txt", "comp": "NaCl", "w_ppt": 1000},
        {"filename": "sample_002.txt", "exclude": true},
        {"filename": "sample_003_test.txt", "exclude": true},
        {"filename": "sample_004.txt", "comp": "NaCl", "w_ppt": 1000}
      ]
    }
  ]
}
```

## Command Line Tools

### Generate Examples
```bash
python calibration_config.py
```

### Scan Data Directory
```bash
# Scan and generate config from your data
python calibration_config.py scan data/20250813

# Output to custom filename
python calibration_config.py scan data/20250813 my_calibration.json
```

This automatically:
- Finds all .txt files
- Identifies KCl files as standards
- Groups everything else as measurements
- Creates a template config you can edit

### Run with Config
```bash
# Auto-detect config in data directory
python gamry_HiPOZOZ.py

# Specify config path
python gamry_HiPOZOZ.py --config path/to/my_config.json

# Multiple data directories
python gamry_HiPOZOZ.py --dates 20250813 20250814
```

## How It Works

### Filename Matching

- Filenames must match **exactly** (case-sensitive)
- Config can use basenames: `"KCl_84.txt"`
- Or full names: `"Default_20250813_KCl_84uScm_P100_T298.txt"`
- Both will match against full paths in timeseries

### Cell Constant Calculation

For each calibration group:

1. **Find standards**: Locate files listed in `"standards"`
2. **Compute K_cell**: For each standard, K = σ × R
3. **Average**: K_cell = mean(K values), with uncertainty
4. **Apply to measurements**: σ = K_cell / R for each measurement

### Multiple Groups

Each group is independent:
- Group 1 standards → Group 1 cell constant → Group 1 measurements
- Group 2 standards → Group 2 cell constant → Group 2 measurements

This handles instrument drift between measurement sessions.

## Troubleshooting

### "Standard file not found"

**Problem:** Config lists a file that doesn't exist or doesn't match.

**Solutions:**
- Check filename spelling (case-sensitive!)
- Use `ls data/20250813/ConductivityData_Default/*.txt` to see actual filenames
- Copy exact filenames into config
- Or use `python calibration_config.py scan data/20250813` to auto-generate

### "No valid cell constants computed"

**Problem:** Standards found but can't compute cell constant.

**Solutions:**
- Ensure standard files have conductivity in their description
- Check that impedance fits succeeded (R values not NaN)
- Verify standards have positive conductivity and resistance

### Config not auto-detected

**Solutions:**
- Place config in data directory (same level as `ConductivityData_Default/`)
- Name it: `calibration_config.json` or `calibration.json`
- Or specify explicitly: `--config path/to/config.json`

### Wrong calibration applied

**Problem:** Measurements using wrong standards.

**Solution:**
- Check your calibration groups
- Ensure measurements are in the correct group with their corresponding standards
- Each group is independent - measurements only use standards from their group

## Benefits of Explicit Filenames

✅ **Precise control**: Specify exactly which files to use
✅ **Bracketing support**: Multiple calibration groups for time-series
✅ **Reproducible**: Exact record of which files were analyzed
✅ **Flexible**: Mix different compositions and conditions
✅ **Traceable**: See filenames in GUI to verify
✅ **No surprises**: No wildcard matching mistakes

## Benefits of Multiple Groups

✅ **Bracketing**: Standards before and after measurements
✅ **Drift correction**: Each session gets its own cell constant
✅ **Long experiments**: Recalibrate periodically
✅ **Multiple compositions**: Different standards for different samples
✅ **Flexibility**: Organize your experiment however you want

## Comparison to Old Pattern-Based Config

**Old way (pattern-based):**
```json
{
  "standards": [{"pattern": "*KCl*"}],
  "measurements": [{"pattern": "*NaCl*"}]
}
```
- ❌ All standards lumped together
- ❌ Can't bracket measurements
- ❌ Wildcards might match wrong files

**New way (explicit filenames + groups):**
```json
{
  "calibrations": [
    {
      "name": "Morning",
      "standards": ["KCl_0800.txt", "KCl_0805.txt"],
      "measurements": ["NaCl_0900.txt"]
    },
    {
      "name": "Afternoon",
      "standards": ["KCl_1400.txt"],
      "measurements": ["NaCl_1500.txt"]
    }
  ]
}
```
- ✅ Precise filename control
- ✅ Supports bracketing
- ✅ Multiple independent groups
- ✅ Each group has its own cell constant

## Advanced: Programmatic Usage

```python
from calibration_config import CalibrationConfig

# Load config
config = CalibrationConfig('my_calibration.json')

# Apply all groups
results = config.apply_to_timeseries(timeseries)

for result in results:
    print(f"{result['name']}: {result['message']}")
    print(f"  K_cell = {result['cell_constant']:.2f} ± {result['cell_constant_unc']:.2f} 1/m")
    print(f"  Standards: {result['standard_indices']}")
    print(f"  Measurements: {result['measurement_indices']}")
```

## Manual GUI Workflow Still Available

Don't want to use config files? The manual workflow still works:

1. Run without config: `python gamry_HiPOZOZ.py`
2. Select standard rows in table (use filename column to identify them!)
3. Click "Mark as Standard"
4. Select measurement rows
5. Click "Associate Measurements"

The config approach automates these steps and supports bracketing!

## See Also

- `CLAUDE.md` - Full project documentation
- `python calibration_config.py --help` - Command line help
