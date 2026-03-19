# Config File Harmonization Command

## Overview

The `--harmonize` command synchronizes CSV and JSON config files, ensuring both formats contain the same data. Use this after editing one format to update the other.

## Why Harmonize?

**Problem:** You edit the CSV in Excel but the JSON is now outdated.

**Solution:** Run harmonize to update the JSON automatically.

**Benefits:**
- ✅ Both formats stay synchronized
- ✅ No manual copying/editing
- ✅ Preserves all fields (notes, compositions, etc.)
- ✅ Works bidirectionally (CSV→JSON or JSON→CSV)

## Usage

### Via gamry_HiPOZOZ.py (Recommended)

```bash
# After editing CSV in Excel:
python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis20250815.csv

# After editing JSON:
python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis20250815.json
```

### Via harmonize_config.py (Standalone)

```bash
# CSV → JSON:
python harmonize_config.py data/20250815/zAnalysis20250815.csv

# JSON → CSV:
python harmonize_config.py data/20250815/zAnalysis20250815.json

# Overwrite existing:
python harmonize_config.py --force data/20250815/zAnalysis20250815.csv

# Quiet mode:
python harmonize_config.py --quiet data/20250815/zAnalysis20250815.csv
```

## Workflow Examples

### Example 1: Edit CSV in Excel

**Steps:**
1. Open `data/20250815/zAnalysis20250815.csv` in Excel
2. Edit standards, measurements, notes, concentrations
3. Save file
4. Run: `python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis20250815.csv`
5. ✓ JSON is now synchronized

**Result:**
```
Reading CSV: data/20250815/zAnalysis20250815.csv
✓ Created JSON: data/20250815/zAnalysis20250815.json
  Groups: 1
  Group 1: 1 standards, 12 measurements
```

### Example 2: Edit JSON Programmatically

**Steps:**
1. Script modifies `data/20250815/zAnalysis20250815.json`
2. Run: `python harmonize_config.py data/20250815/zAnalysis20250815.json`
3. ✓ CSV is now synchronized

**Result:**
```
Reading JSON: data/20250815/zAnalysis20250815.json
✓ Created CSV: data/20250815/zAnalysis20250815.csv
  Total rows: 13
  Group 1: 1 standards, 12 measurements
```

### Example 3: Batch Harmonize All Configs

```bash
# Harmonize all CSV files in data directories:
for f in data/*/zAnalysis*.csv; do
    python harmonize_config.py "$f"
done

# Or with gamry_HiPOZOZ.py:
for f in data/*/zAnalysis*.csv; do
    python gamry_HiPOZOZ.py --harmonize "$f"
done
```

## What Gets Synchronized

### From CSV to JSON
- ✅ group_name → calibration group structure
- ✅ filename
- ✅ P_MPa, T_K → numeric or string
- ✅ type (standard/measurement)
- ✅ conductivity_Sm (for standards)
- ✅ comp (composition)
- ✅ w_ppt, w_molal (concentrations)
- ✅ exclude → boolean true/false
- ✅ notes → preserved exactly

### From JSON to CSV
- ✅ calibration_groups → group_name rows
- ✅ standards → type=standard rows
- ✅ measurements → type=measurement rows
- ✅ All metadata fields preserved
- ✅ exclude: true → 'x' in CSV

## Field Handling

### Numeric Fields (P_MPa, T_K, conductivity_Sm)

**CSV:**
```csv
P_MPa,T_K,conductivity_Sm
0.1,292,8
```

**JSON:**
```json
{
  "P_MPa": 0.1,
  "T_K": 292,
  "conductivity_Sm": 8.0
}
```

### Exclude Field

**CSV:**
```csv
exclude
x
yes
true
1
```

**JSON:**
```json
{
  "exclude": true
}
```

Any of `x`, `yes`, `true`, or `1` in CSV becomes `true` in JSON.
Empty in CSV = `false` in JSON (field omitted).

### Notes Field

**CSV:**
```csv
notes
outside pressure vessel but software was reading P and T from that system
```

**JSON:**
```json
{
  "notes": "outside pressure vessel but software was reading P and T from that system"
}
```

Notes are preserved exactly, including commas and special characters.

## Verification

### Check CSV→JSON Conversion

```bash
# Convert
python harmonize_config.py data/20250815/zAnalysis20250815.csv

# Verify
cat data/20250815/zAnalysis20250815.json | head -30
```

### Check JSON→CSV Conversion

```bash
# Convert
python harmonize_config.py data/20250815/zAnalysis20250815.json

# Verify
head -20 data/20250815/zAnalysis20250815.csv
```

### Compare Before/After

```bash
# Backup original
cp data/20250815/zAnalysis20250815.json backup.json

# Edit CSV in Excel, save

# Harmonize
python harmonize_config.py data/20250815/zAnalysis20250815.csv

# Compare
diff backup.json data/20250815/zAnalysis20250815.json
```

## Common Use Cases

### Case 1: Student Edits CSV in Excel

**Scenario:** Student creates config in Excel, needs JSON for automation

**Command:**
```bash
python gamry_HiPOZOZ.py --harmonize data/StudentData/zAnalysis.csv
```

**Result:** JSON created with identical data

### Case 2: Script Generates JSON

**Scenario:** Python script creates JSON config, instructor wants CSV for review

**Command:**
```bash
python harmonize_config.py data/ScriptOutput/config.json
```

**Result:** CSV created for Excel viewing

### Case 3: Fix Desynchronized Files

**Scenario:** CSV and JSON both exist but have different data

**Steps:**
1. Determine which is correct (CSV or JSON)
2. Delete the incorrect one
3. Run harmonize on the correct one
4. Both files now match

**Example:**
```bash
# CSV is correct, JSON is outdated
rm data/20250815/zAnalysis20250815.json
python harmonize_config.py data/20250815/zAnalysis20250815.csv
# ✓ Fresh JSON created from CSV
```

### Case 4: Add Notes After Initial Analysis

**Scenario:** Analysis already run, need to add notes to config

**Steps:**
1. Open CSV in Excel
2. Add notes in "notes" column
3. Save
4. Harmonize: `python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis.csv`
5. Re-run analysis with updated config

## Error Handling

### File Not Found

```bash
$ python harmonize_config.py data/nonexistent/config.csv
ERROR: File not found: data/nonexistent/config.csv
```

### File Already Exists

```bash
$ python harmonize_config.py data/20250815/zAnalysis.csv
Reading CSV: data/20250815/zAnalysis.csv
WARNING: JSON file already exists: data/20250815/zAnalysis.json
Use --force to overwrite
```

**Solution:** Use `--force` flag or via gamry_HiPOZOZ.py (always forces):
```bash
python harmonize_config.py --force data/20250815/zAnalysis.csv
# or
python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis.csv
```

### Invalid CSV Format

**Problem:** Missing required columns

**Fix:** Ensure CSV has these columns:
```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
```

### Invalid JSON Format

**Problem:** Malformed JSON syntax

**Fix:** Validate JSON:
```bash
python -m json.tool data/20250815/zAnalysis.json
```

## Integration with Workflow

### GUI Workflow

1. Run analysis: `python gamry_HiPOZOZ.py --dates 20250815`
2. GUI creates initial config files (both CSV and JSON)
3. Edit CSV in Excel to add notes/metadata
4. Harmonize: `python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis.csv`
5. Re-run analysis with updated config

**Note:** GUI automatically harmonizes on save, but manual harmonize ensures consistency.

### Headless Workflow

1. Create/edit CSV config in Excel
2. Harmonize: `python gamry_HiPOZOZ.py --harmonize data/20250815/zAnalysis.csv`
3. Run headless: `python gamry_HiPOZOZ.py --headless --dates 20250815`
4. Results saved to same directory

### Batch Processing Workflow

```bash
# For each student directory:
for dir in data/Student*; do
    # Create CSV template
    python calibration_config.py --generate "$dir"

    # Student edits CSV in Excel
    # (pause for student to edit)

    # Harmonize to create JSON
    python harmonize_config.py "$dir"/zAnalysis*.csv

    # Run analysis
    python gamry_HiPOZOZ.py --headless --dates $(basename "$dir")
done
```

## Command Line Reference

### harmonize_config.py

```
usage: harmonize_config.py [-h] [-f] [-q] FILE

Harmonize calibration config files between CSV and JSON formats

positional arguments:
  FILE           Path to CSV or JSON config file to harmonize

optional arguments:
  -h, --help     show this help message and exit
  -f, --force    Overwrite existing matching file
  -q, --quiet    Quiet mode - minimal output
```

### gamry_HiPOZOZ.py --harmonize

```
usage: gamry_HiPOZOZ.py [--harmonize FILE] [other options]

optional arguments:
  --harmonize FILE  Harmonize CSV↔JSON config file (creates matching format) and exit
```

**Differences:**
- `gamry_HiPOZOZ.py --harmonize` always uses `--force` (overwrites)
- `harmonize_config.py` requires `--force` flag to overwrite
- Both produce identical results

## Tips

### Tip 1: Always Harmonize After Excel Edits

Make it a habit:
```bash
# Edit CSV in Excel
# Save and close Excel
python gamry_HiPOZOZ.py --harmonize data/YourData/zAnalysis.csv
```

### Tip 2: Use CSV as Primary Format

**Recommendation:** Edit CSV in Excel, harmonize to create JSON

**Reason:**
- Excel is familiar to students
- Visual table editing is easier
- Notes field handles commas/quotes better in CSV

### Tip 3: Verify After Harmonization

```bash
# Harmonize
python harmonize_config.py data/20250815/zAnalysis.csv

# Quick check
wc -l data/20250815/zAnalysis.csv
# Should match number of standards + measurements + 1 header
```

### Tip 4: Backup Before Forcing

```bash
# Backup existing files
cp data/20250815/zAnalysis.json data/20250815/zAnalysis.json.backup

# Harmonize with force
python harmonize_config.py --force data/20250815/zAnalysis.csv

# If something wrong, restore:
# mv data/20250815/zAnalysis.json.backup data/20250815/zAnalysis.json
```

## Troubleshooting

### Problem: Notes are truncated

**Cause:** Commas in notes field not quoted in CSV

**Solution:** Excel automatically handles this when saving. If editing manually:
```csv
notes
"Text with, commas, in it"
```

### Problem: Numbers become dates in Excel

**Example:** `0.1` becomes `Jan 0`

**Solution:** Format cells as "Number" or "Text" before entering data:
1. Select column
2. Format → Cells → Number
3. Enter values

### Problem: Can't find harmonize_config.py

**Cause:** Running from wrong directory

**Solution:**
```bash
cd /path/to/hipozgenai
python harmonize_config.py data/20250815/zAnalysis.csv
```

### Problem: Permission denied

**Cause:** File is open in Excel

**Solution:** Close Excel, then harmonize

## See Also

- `FORMAT_HARMONIZATION.md` - CSV/JSON format specifications
- `DIRECTORY_SELECTION_UPDATE.md` - Directory selection workflow
- `calibration_config.py` - Config file generation and loading
