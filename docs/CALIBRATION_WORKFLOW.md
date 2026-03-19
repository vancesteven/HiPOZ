# HiPOZ Calibration Workflow Guide

## Which Format to Use?

### **Recommendation: Use JSON** ✅

**JSON is the primary/main format** because:
- More flexible and readable
- Supports comments (in the "notes" field)
- Easier to organize multiple calibration groups
- Better for version control (git diffs)

**CSV is available** for simple cases or if you prefer spreadsheet editing.

## Standard Calibration Procedure

### Step 1: Identify Your Standards

List the KCl standard files you measured. These have **known conductivity values**.

Common KCl standards:
- **84 µS/cm** = 0.0084 S/m
- **447 µS/cm** = 0.0447 S/m
- **2070 µS/cm** = 0.2070 S/m
- **15000 µS/cm** = 1.5 S/m

### Step 2: Create Config File

You have **two options** for specifying conductivity:

#### **Option A: Specify Conductivity in Config (Recommended)** ✅

Explicitly assign conductivity values in the config file:

**JSON format:**
```json
{
  "calibrations": [
    {
      "name": "My experiment",
      "standards": [
        {"filename": "KCl_84_morning.txt", "conductivity_Sm": 0.0084},
        {"filename": "KCl_447_morning.txt", "conductivity_Sm": 0.0447}
      ],
      "measurements": [
        "NaCl_sample1.txt",
        "NaCl_sample2.txt"
      ]
    }
  ]
}
```

**CSV format:**
```csv
group_name,filename,type,conductivity_Sm,notes
Morning,KCl_84_morning.txt,standard,0.0084,84 uS/cm standard
Morning,KCl_447_morning.txt,standard,0.0447,447 uS/cm standard
Morning,NaCl_sample1.txt,measurement,,Sample at 100 MPa
Morning,NaCl_sample2.txt,measurement,,Sample at 150 MPa
```

#### **Option B: Read from File Metadata**

If your Gamry files already have conductivity in the description field, just list the filename:

**JSON format:**
```json
{
  "calibrations": [
    {
      "name": "My experiment",
      "standards": [
        "KCl_84_morning.txt",
        "KCl_447_morning.txt"
      ],
      "measurements": [...]
    }
  ]
}
```

**CSV format:**
```csv
group_name,filename,type,conductivity_Sm,notes
Morning,KCl_84_morning.txt,standard,,Reads from file
Morning,KCl_447_morning.txt,standard,,Reads from file
```

### Step 3: Organize into Groups (for Bracketing)

If you measured standards at different times (before/after measurements), create **multiple calibration groups**:

```json
{
  "calibrations": [
    {
      "name": "Before lunch",
      "standards": [
        {"filename": "KCl_84_0900.txt", "conductivity_Sm": 0.0084},
        {"filename": "KCl_447_0900.txt", "conductivity_Sm": 0.0447}
      ],
      "measurements": [
        "NaCl_1000.txt",
        "NaCl_1100.txt"
      ]
    },
    {
      "name": "After lunch",
      "standards": [
        {"filename": "KCl_84_1400.txt", "conductivity_Sm": 0.0084},
        {"filename": "KCl_447_1400.txt", "conductivity_Sm": 0.0447}
      ],
      "measurements": [
        "NaCl_1500.txt",
        "NaCl_1600.txt"
      ]
    }
  ]
}
```

Each group gets its own cell constant, accounting for drift!

### Step 4: Save Config File

Save as:
- `data/20250813/calibration_config.json` (auto-detected)
- Or any path, then specify: `python gamry_HiPOZOZ.py --config my_config.json`

### Step 5: Run HiPOZ

```bash
python gamry_HiPOZOZ.py
```

The system will:
1. ✅ Load your config
2. ✅ For each group:
   - Calculate cell constant: K = σ × R (averaged across standards)
   - Apply to measurements: σ = K / R
3. ✅ Show summary in GUI
4. ✅ Pre-fill table with results
5. ✅ Generate S vs P plot
6. ✅ Save to hipoz_exports/

## Quick Start Template

### JSON Template (Main Format)

```json
{
  "description": "Replace with your experiment name",
  "calibrations": [
    {
      "name": "Replace with session name",
      "standards": [
        {"filename": "your_KCl_file1.txt", "conductivity_Sm": 0.0084},
        {"filename": "your_KCl_file2.txt", "conductivity_Sm": 0.0447}
      ],
      "measurements": [
        "your_sample_file1.txt",
        "your_sample_file2.txt"
      ]
    }
  ]
}
```

**To use:**
1. Copy template
2. Replace filenames with your actual files
3. Set conductivity values (in S/m)
4. Save as `calibration_config.json`
5. Run `python gamry_HiPOZOZ.py`

### CSV Template (Alternative)

```csv
group_name,filename,type,conductivity_Sm,notes
Group1,KCl_84.txt,standard,0.0084,
Group1,KCl_447.txt,standard,0.0447,
Group1,sample1.txt,measurement,,
Group1,sample2.txt,measurement,,
```

## Common KCl Standards - Conductivity Reference

| Label (µS/cm) | Conductivity (S/m) | JSON Value |
|---------------|--------------------| ---------- |
| 84 | 0.0084 | `0.0084` |
| 447 | 0.0447 | `0.0447` |
| 2070 | 0.2070 | `0.2070` |
| 2764 | 0.2764 | `0.2764` |
| 15000 | 1.5 | `1.5` |
| 80000 | 8.0 | `8.0` |

**Conversion:** µS/cm ÷ 10000 = S/m

Examples:
- 84 µS/cm ÷ 10000 = 0.0084 S/m
- 447 µS/cm ÷ 10000 = 0.0447 S/m

## Example Workflows

### Example 1: Simple Experiment (Single Group)

**Scenario:** You measured 2 standards, then 5 samples.

**Config:**
```json
{
  "calibrations": [
    {
      "name": "Full experiment",
      "standards": [
        {"filename": "KCl_84.txt", "conductivity_Sm": 0.0084},
        {"filename": "KCl_447.txt", "conductivity_Sm": 0.0447}
      ],
      "measurements": [
        "NaCl_P100.txt",
        "NaCl_P150.txt",
        "NaCl_P200.txt",
        "NaCl_P250.txt",
        "NaCl_P300.txt"
      ]
    }
  ]
}
```

### Example 2: Bracketed Measurements (Multiple Groups)

**Scenario:** Standards before and after lunch.

**Config:**
```json
{
  "calibrations": [
    {
      "name": "Morning",
      "standards": [
        {"filename": "KCl_84_AM.txt", "conductivity_Sm": 0.0084}
      ],
      "measurements": [
        "Sample_1000.txt",
        "Sample_1030.txt",
        "Sample_1100.txt"
      ]
    },
    {
      "name": "Afternoon",
      "standards": [
        {"filename": "KCl_84_PM.txt", "conductivity_Sm": 0.0084}
      ],
      "measurements": [
        "Sample_1300.txt",
        "Sample_1330.txt",
        "Sample_1400.txt"
      ]
    }
  ]
}
```

### Example 3: Multiple Standards per Group

**Scenario:** You want to average across 3 different KCl standards.

**Config:**
```json
{
  "calibrations": [
    {
      "name": "Calibration set 1",
      "standards": [
        {"filename": "KCl_84.txt", "conductivity_Sm": 0.0084},
        {"filename": "KCl_447.txt", "conductivity_Sm": 0.0447},
        {"filename": "KCl_2070.txt", "conductivity_Sm": 0.2070}
      ],
      "measurements": [
        "Sample1.txt",
        "Sample2.txt"
      ]
    }
  ]
}
```

The cell constant will be averaged across all 3 standards.

## Workflow Summary

```
1. Measure standards (KCl at known conductivity)
2. Measure samples (unknown conductivity)
3. Create config file:
   - List standard filenames + conductivity values
   - List measurement filenames
   - Group into multiple calibrations if bracketing
4. Save config in data directory
5. Run: python gamry_HiPOZOZ.py
6. Results appear in GUI automatically!
```

## Both Formats Side-by-Side

### Same Config in JSON and CSV

**JSON (Recommended):**
```json
{
  "calibrations": [
    {
      "name": "Session1",
      "standards": [
        {"filename": "std1.txt", "conductivity_Sm": 0.0084},
        {"filename": "std2.txt", "conductivity_Sm": 0.0447}
      ],
      "measurements": ["meas1.txt", "meas2.txt"]
    }
  ]
}
```

**CSV (Alternative):**
```csv
group_name,filename,type,conductivity_Sm,notes
Session1,std1.txt,standard,0.0084,
Session1,std2.txt,standard,0.0447,
Session1,meas1.txt,measurement,,
Session1,meas2.txt,measurement,,
```

## Auto-Generate Starting Point

Don't want to type all filenames? Generate a template:

```bash
python calibration_config.py scan data/20250813
```

This creates `calibration_config.json` with all your files listed. Then edit it to:
- Add conductivity values to standards
- Split into multiple groups if needed
- Remove any unwanted files

## Tips

✅ **Use JSON** as your primary format
✅ **Specify conductivity explicitly** in config (easier to verify)
✅ **Create multiple groups** for bracketing (accounts for drift)
✅ **Average multiple standards** for better cell constant
✅ **Name your groups meaningfully** (e.g., "Before lunch", "After lunch")

## Questions?

- See `CALIBRATION_README.md` for detailed documentation
- See `CLAUDE.md` for full project documentation
- Run `python calibration_config.py` to generate examples
