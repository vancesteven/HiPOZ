# Format Harmonization: CSV, JSON, and GUI

## Overview

HiPOZ maintains format consistency across CSV, JSON, and GUI operations. When you work with a CSV config, the system stays in CSV. When you work with JSON, it stays in JSON.

## Key Principles

### 1. **Format Preservation**
- **Load CSV → Save CSV**: If you load `zAnalysis20250813.csv`, the GUI saves back to CSV
- **Load JSON → Save JSON**: If you load `zAnalysis20250813.json`, the GUI saves back to JSON
- **No Cross-Contamination**: CSV edits don't overwrite JSON, and vice versa

### 2. **Directory Organization**
All config files for a date are stored together:
```
data/
  20250813/
    ConductivityData_Default/
      *.txt                        # Measurement files
    zAnalysis20250813.csv          # CSV config (Excel-friendly)
    zAnalysis20250813.json         # JSON config (optional)
```

### 3. **Priority System**
When multiple formats exist, the system uses this priority:
1. **CSV first** (preferred for Excel users)
2. **JSON second** (for advanced users)
3. **Legacy names** (backward compatibility)

## Workflows

### Workflow A: CSV-Only (Recommended for Students)

```bash
# 1. Generate CSV config
python calibration_config.py scan data/20250813

# 2. Move to data directory
mv zAnalysis20250813.csv data/20250813/

# 3. Edit in Excel
# Open data/20250813/zAnalysis20250813.csv
# - Set type (standard/measurement)
# - Add conductivity_Sm for standards
# - Add comp and w_ppt for measurements
# - Mark exclude='x' for bad files

# 4. Run analysis
python gamry_HiPOZOZ.py

# 5. GUI processes data
# - Loads CSV
# - Applies calibration
# - Saves results back to CSV
# - CSV stays synchronized
```

**Result**: Only CSV file is used and modified. No JSON created.

### Workflow B: JSON-Only (Advanced Users)

```bash
# 1. Generate JSON config
python calibration_config.py scan data/20250813 --format json

# 2. Move to data directory
mv zAnalysis20250813.json data/20250813/

# 3. Edit JSON file
# - Add conductivity_Sm, comp, w_ppt
# - Add "exclude": true for bad files
# - Create multiple calibration groups

# 4. Run analysis
python gamry_HiPOZOZ.py

# 5. GUI processes data
# - Loads JSON
# - Applies calibration
# - Saves results back to JSON
# - JSON stays synchronized
```

**Result**: Only JSON file is used and modified. No CSV created.

### Workflow C: Both Formats (Flexibility)

You can maintain both CSV and JSON in the same directory:

```
data/20250813/
  zAnalysis20250813.csv    # Used by Excel users
  zAnalysis20250813.json   # Used by advanced users
```

**How it works:**
- System loads CSV by default (higher priority)
- GUI saves back to CSV only
- JSON remains unchanged unless you explicitly load it
- Each format is independent

**When to use both:**
- Team has mix of Excel and text editor users
- Want backup in different format
- Migrating from JSON to CSV workflow

## GUI Behavior

### On Startup

```python
# GUI checks for config files:
1. Look for: data/20250813/zAnalysis20250813.csv
   - If found: Load CSV → Will save to CSV

2. If not found, look for: data/20250813/zAnalysis20250813.json
   - If found: Load JSON → Will save to JSON

3. If neither found:
   - Auto-create: zAnalysis20250813.csv (Excel-friendly default)
   - Will save to CSV
```

### During Analysis

When you mark standards or associate measurements:
```python
# Mark as Standard button clicked
→ GUI updates internal state
→ Saves to: self.calib_config.config_path
→ Format: Same as loaded (CSV or JSON)
```

### On Save

The `save_gui_state_to_config()` method:
1. Detects format from file extension
2. Calls appropriate saver:
   - `_save_to_csv()` for CSV files
   - `_save_to_json()` for JSON files
3. Preserves existing structure (groups, notes)
4. Adds timestamp to track updates

## File Synchronization

### What Gets Synchronized

When GUI saves:
- ✅ Standards list (with conductivity_Sm, comp, w_ppt)
- ✅ Measurements list (with comp, w_ppt)
- ✅ Calibration groups
- ✅ Timestamp of last update

### What Stays Independent

- ✅ Format (CSV ≠ JSON)
- ✅ Comments/notes (format-specific)
- ✅ File location (must be in data/<date>/)

## Format Comparison

| Feature | CSV | JSON |
|---------|-----|------|
| **Excel editing** | ✅ Native | ❌ Text editor |
| **Student-friendly** | ✅ Familiar | ❌ Less familiar |
| **Exclude files** | `x` in column | `"exclude": true` |
| **Multiple groups** | Rows with group_name | Nested arrays |
| **GUI saves to** | Same CSV | Same JSON |
| **Priority** | 1st (default) | 2nd (fallback) |

## Testing

Run the harmonization test suite:
```bash
python test_format_harmonization.py
```

This verifies:
- ✅ CSV configs stay in CSV format
- ✅ JSON configs stay in JSON format
- ✅ All configs stored in data/<date>/ directory
- ✅ Exclude feature working correctly
- ✅ GUI saves to correct format

## Best Practices

### For Students (Excel Users)
1. **Use CSV only** - it's simpler
2. Generate with: `python calibration_config.py scan data/<date>`
3. Edit in Excel - familiar spreadsheet interface
4. Mark bad files with `x` in exclude column
5. Run analysis - GUI stays in CSV

### For Advanced Users
1. **Choose your format** - CSV or JSON
2. Stick with one format to avoid confusion
3. If you have both, CSV takes priority
4. JSON gives more control over structure
5. Both formats fully supported

### For Mixed Teams
1. **Standardize on CSV** - easier for everyone
2. Keep JSON for advanced configurations
3. CSV takes priority when both exist
4. Document which format your team uses
5. Use exclude column for bad data

## Troubleshooting

**"Which file is being used?"**
- Check the log output: `[INFO] Found CSV analysis config: ...`
- CSV has priority over JSON
- GUI saves to whichever format was loaded

**"I edited CSV but GUI loaded JSON"**
- Check file names - must be `zAnalysis<date>.csv`
- Check location - must be in `data/<date>/`
- CSV takes priority if both exist

**"GUI created JSON, I want CSV"**
- Delete the JSON: `rm data/<date>/zAnalysis<date>.json`
- Create CSV: `python calibration_config.py scan data/<date>`
- Move to directory: `mv zAnalysis<date>.csv data/<date>/`

**"Can I convert between formats?"**
- No automatic converter yet
- Manual: Open CSV in Excel, copy data to JSON structure
- Or regenerate: `python calibration_config.py scan data/<date> --format csv`

## Summary

✅ **Format is preserved**: CSV → CSV, JSON → JSON
✅ **No cross-contamination**: Each format independent
✅ **Same directory**: All configs in data/<date>/
✅ **Priority system**: CSV first, JSON second
✅ **GUI synchronization**: Saves to loaded format
✅ **Exclude support**: Both formats support exclusion
✅ **Student-friendly**: CSV recommended for Excel users
