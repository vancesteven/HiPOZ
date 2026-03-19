# Format Harmonization - Implementation Summary

## ✅ What's Been Implemented

### 1. **Format Preservation System**

The system now ensures CSV and JSON formats stay synchronized with the GUI:

- **Load CSV → Save CSV**: When you load a CSV config, the GUI always saves back to CSV
- **Load JSON → Save JSON**: When you load a JSON config, the GUI always saves back to JSON
- **No Cross-Contamination**: Editing CSV doesn't affect JSON, and vice versa

### 2. **Code Changes**

#### `DataSelector.py` Updates:
- `save_gui_state_to_config()` now detects file format and saves appropriately
- New `_save_to_csv()` method for CSV format saving
- New `_save_to_json()` method for JSON format saving
- Tracks loaded config path from `calib_config.config_path`
- Preserves format from loaded file

#### `calibration_config.py` Updates:
- Stores `config_path` in CalibrationConfig class
- CSV format is default for auto-generation
- Exclude support in both CSV and JSON
- Format flag: `--format csv` or `--format json`

#### `gamry_HiP.py` Updates:
- CSV has priority over JSON when both exist
- Auto-detects config files in correct order

### 3. **Directory Structure**

All config files stored in the same location:
```
data/20250813/
  ├── ConductivityData_Default/
  │   └── *.txt                        # Measurement files
  ├── zAnalysis20250813.csv           # CSV config (Excel-friendly)
  └── zAnalysis20250813.json          # JSON config (optional)
```

### 4. **Testing Results**

All tests pass ✅:

```
✓ CSV Format Harmonization
  - Loaded CSV from: data/20250813/zAnalysis20250813.csv
  - Will save to: data/20250813/zAnalysis20250813.csv
  - Format preserved: CSV

✓ JSON Format Harmonization
  - Loaded JSON from: data/20250813/zAnalysis20250813.json
  - Will save to: data/20250813/zAnalysis20250813.json
  - Format preserved: JSON

✓ Directory Structure
  - Both files in: data/20250813/
  - CSV priority when both exist

✓ Exclude Feature
  - Works in both CSV and JSON
  - Files marked with 'x' or 'exclude': true are skipped
  - 77 files loaded (78 total - 1 excluded)
```

## 📋 Usage Examples

### Example 1: CSV Workflow (Students)

```bash
# 1. Generate CSV config
python calibration_config.py scan data/20250813

# 2. Move to data directory
mv zAnalysis20250813.csv data/20250813/

# 3. Edit in Excel
# - Set type=standard or type=measurement
# - Add conductivity_Sm for standards (e.g., 0.0084)
# - Add comp and w_ppt for measurements
# - Mark exclude='x' for bad files

# 4. Run GUI
python gamry_HiP.py
# → Loads CSV
# → Saves back to CSV
# → No JSON created
```

### Example 2: JSON Workflow (Advanced)

```bash
# 1. Generate JSON config
python calibration_config.py scan data/20250813 --format json

# 2. Move to data directory
mv zAnalysis20250813.json data/20250813/

# 3. Edit JSON
# - Add conductivity_Sm, comp, w_ppt
# - Add "exclude": true for bad files

# 4. Run GUI
python gamry_HiP.py
# → Loads JSON
# → Saves back to JSON
# → No CSV created
```

### Example 3: Both Formats (Flexibility)

```bash
# You can have both:
data/20250813/
  ├── zAnalysis20250813.csv    # For Excel users
  └── zAnalysis20250813.json   # For advanced users

# When you run:
python gamry_HiP.py
# → CSV loads (priority)
# → Saves to CSV only
# → JSON unchanged
```

## 🔧 Technical Details

### Save Logic Flow

```python
# In DataSelector.save_gui_state_to_config():

1. Determine config file path
   - First check: self.calib_config.config_path
   - Then check: self.config_file_paths[date]

2. Detect format from extension
   - .csv → Call _save_to_csv()
   - .json → Call _save_to_json()

3. Save in native format
   - CSV: Write rows with headers
   - JSON: Write nested structure
```

### CSV Save Format

```csv
group_name,filename,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
Group 1,std1.txt,standard,0.0084,KCl,1.0,,,Updated by GUI
Group 1,meas1.txt,measurement,,NaCl,1000,,,Updated by GUI
```

### JSON Save Format

```json
{
  "description": "Analysis for 20250813",
  "calibrations": [{
    "name": "Group 1",
    "standards": [
      {"filename": "std1.txt", "conductivity_Sm": 0.0084, "comp": "KCl", "w_ppt": 1.0}
    ],
    "measurements": [
      {"filename": "meas1.txt", "comp": "NaCl", "w_ppt": 1000}
    ]
  }],
  "notes": ["Updated from GUI at 2026-03-18 16:00:00"]
}
```

## 📚 Documentation

New/Updated Documentation:
- ✅ `FORMAT_HARMONIZATION.md` - Complete harmonization guide
- ✅ `QUICK_START.md` - Updated with format info
- ✅ `CALIBRATION_README.md` - Updated with CSV/JSON examples
- ✅ `MEMORY.md` - Updated with harmonization notes
- ✅ `test_format_harmonization.py` - Test suite

## ✨ Key Benefits

1. **No Confusion**: Format you load is format you save
2. **Student-Friendly**: CSV recommended for Excel users
3. **Flexible**: Advanced users can use JSON
4. **Safe**: No accidental overwriting between formats
5. **Organized**: All configs in `data/<date>/` directory
6. **Tested**: Full test suite verifies behavior
7. **Documented**: Comprehensive documentation

## 🎯 Recommendations

### For Students
- **Use CSV** - easier in Excel
- Generate with: `python calibration_config.py scan data/<date>`
- Edit in Excel
- Mark bad files with 'x'

### For Researchers
- **Use CSV** for quick edits
- **Use JSON** for complex configurations
- Pick one format and stick with it
- Both formats fully supported

### For Mixed Teams
- **Standardize on CSV** - easier for everyone
- CSV has priority when both exist
- Document which format your team uses

## 🔍 Testing

Run the test suite:
```bash
python test_format_harmonization.py
```

Expected output:
```
✓ ALL TESTS PASSED

Summary:
  • CSV configs stay in CSV format
  • JSON configs stay in JSON format
  • All configs stored in data/<date>/ directory
  • Exclude feature working correctly
  • GUI will save to the same format that was loaded
```

## 📝 Next Steps

The system is fully functional and tested. You can now:

1. Generate config files (CSV or JSON)
2. Edit in your preferred format
3. Run the GUI
4. GUI saves back to same format
5. Repeat as needed

All formats harmonized and synchronized! ✅
