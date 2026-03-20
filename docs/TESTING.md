# HiPOZ Testing Guide

Comprehensive testing documentation for HiPOZ impedance analyzer.

## Quick Test

Run all tests:
```bash
python test_format_harmonization.py
python test_conductivity_writeback.py
python test_mahboub_analysis.py
cd mahboub2026 && python test_error_handling.py
```

All tests should pass with `✓ ALL TESTS PASSED` or `🎉 All tests passed!` message.

## Test Suites

### 1. Format Harmonization Tests

**File:** `test_format_harmonization.py`

**What it tests:**
- CSV configs remain in CSV format
- JSON configs remain in JSON format
- Both formats stored in same directory
- Exclude feature works correctly
- Files marked with `exclude=x` are skipped

**Run:**
```bash
python test_format_harmonization.py
```

**Expected output:**
```
=== Test 1: CSV Format Harmonization ===
✓ Loaded config from: data/20250813/zAnalysis20250813.csv
  Format: CSV
  Groups: 1
  Group 1: 1 standards, 12 measurements

=== Test 2: JSON Format Harmonization ===
✓ Loaded config from: data/20250813/zAnalysis20250813.json
  Format: JSON
  Groups: 1
  Group 1: 1 standards, 12 measurements

=== Test 3: Directory Structure ===
Data directory: data/20250813
CSV configs: 1
JSON configs: 1

=== Test 4: Exclude Feature ===
Total rows in CSV: 13
Excluded rows: 0
Files in loaded config: 13

✓ ALL TESTS PASSED
```

**What it verifies:**
- ✅ Config files maintain their format (CSV stays CSV, JSON stays JSON)
- ✅ GUI will save back to the same format that was loaded
- ✅ Excluded files are properly filtered out
- ✅ All configs organized in data/<date>/ directories

---

### 2. Conductivity Write-Back Tests

**File:** `test_conductivity_writeback.py`

**What it tests:**
- Computed conductivities written to CSV config
- Computed conductivities written to JSON config
- Both formats updated simultaneously
- Existing metadata preserved (P, T, comp, notes)

**Run:**
```bash
python test_conductivity_writeback.py
```

**Expected output:**
```
=== Test 1: Write-Back to CSV Config ===
✓ CSV file updated: test_config.csv
  ✓ measurement_001.txt: σ = 0.125 S/m
  ✓ measurement_002.txt: σ = 0.098 S/m
  ✓ measurement_003.txt: σ = 0.073 S/m

=== Test 2: Write-Back to JSON Config ===
✓ JSON file updated: test_config.json
  ✓ measurement_001.txt: σ = 0.145 S/m
  ✓ measurement_002.txt: σ = 0.132 S/m

=== Test 3: Both Formats Updated ===
✓ Both files created:
  - test_config.csv
  - test_config.json
✓ Conductivity values match in both formats: 0.156 S/m

=== Test 4: Preserve Existing Metadata ===
✓ Verifying preserved metadata:
  ✓ P_MPa: 25.5
  ✓ T_K: 295.3
  ✓ comp: NaCl
  ✓ w_ppt: 35000
  ✓ w_molal: 0.605
  ✓ notes: Seawater analog, took 3 tries to stabilize
  ✓ conductivity_Sm: 4.567 (newly added)

✓ ALL TESTS PASSED
```

**What it verifies:**
- ✅ Headless analysis writes computed σ values back to original config
- ✅ Both CSV and JSON updated when either is source
- ✅ User notes, P, T, composition preserved
- ✅ Config files remain reproducible and self-documenting

---

### 3. Mahboub Analysis Tests

**File:** `test_mahboub_analysis.py`

**What it tests:**
- Complete analysis workflow for real data
- Config file detection and loading
- Standards have conductivity values
- Measurements have composition metadata
- P_MPa and T_K columns present

**Run:**
```bash
python test_mahboub_analysis.py
```

**Expected output:**
```
========================================
HIPOZ ANALYSIS TEST: 20250813Mahboub2026
========================================

Step 1: Checking data directory...
✓ Data directory found: data/20250813Mahboub2026
✓ Found 13 measurement files

Step 2: Checking config file...
✓ Config found: data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv
  Format: CSV

Step 3: Loading configuration...
✓ Config loaded successfully
  Groups: 1
  Group 1: 1 standards, 12 measurements

Step 4: Analyzing standards...
✓ Found 1 standard(s)
  1. Default_..._KCl.txt
     Conductivity: 0.0084 S/m
     Composition: KCl
✓ 1 standard(s) have conductivity values

Step 5: Analyzing measurements...
✓ Found 12 measurement(s)
  By composition:
    NaCl: 12 measurement(s)

Step 6: Verifying CSV format...
✓ CSV has 11 columns
✓ CSV has 13 rows
✓ CSV includes P_MPa and T_K columns

========================================
ANALYSIS SUMMARY
========================================

✓ Config file: data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv
✓ Format: CSV with P_MPa and T_K columns
✓ Total standards: 1
✓ Total measurements: 12
✓ Total files: 13

✓ READY FOR GUI ANALYSIS
  Run: python gamry_HiPOZ.py
```

**What it verifies:**
- ✅ Real data directory structure is correct
- ✅ Config file auto-detected in data directory
- ✅ Standards have known conductivity values
- ✅ Measurements have composition metadata
- ✅ P and T values properly captured
- ✅ Ready for GUI or headless analysis

---

### 4. Error Handling Tests (Mahboub 2026)

**File:** `mahboub2026/test_error_handling.py`

**What it tests:**
- Missing `conductivity_Sm` column detection (standards not defined)
- All NaN conductivity values detection (standards not associated)
- Partial missing conductivity handling (some measurements missing)
- Concentration grouping and averaging with replicates
- Compound-specific filtering of invalid data

**Run:**
```bash
cd mahboub2026
python test_error_handling.py
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║           Gamry Impedance Data - Error Handling Test Suite         ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

Testing mahboub2026_plots.py error detection and handling...

======================================================================
Test 1: Missing conductivity_Sm column
======================================================================
Scenario: Standards not defined in calibration config

✅ PASS: Correctly detected missing conductivity_Sm column
   Expected behavior: Set gamry_df = None, skip all overlays
   User action: Define standards in zAnalysis config

======================================================================
Test 2: All conductivity values are NaN
======================================================================
Scenario: Standards defined but not associated with measurements

✅ PASS: Correctly detected all NaN conductivity values
   Expected behavior: Set gamry_df = None, skip all overlays
   User action: Associate measurements with standards in HiPOZ GUI

======================================================================
Test 3: Partial missing conductivity
======================================================================
Scenario: Some measurements have conductivity, others don't

✅ PASS: Correctly detected partial missing data (1/4 missing)
   Expected behavior: Show warning, continue with valid data

Testing compound-specific filtering:
  ✅ NaCl: Correctly filtered to 1/2 valid measurements
  ✅ MgSO4: Correctly kept 2/2 valid measurements

   User action: Associate remaining measurements with standards

======================================================================
Test 4: Concentration grouping and averaging
======================================================================
Scenario: Multiple measurements at same concentration should be averaged

✅ PASS: Correctly grouped into 2 concentration bins
  ✅ 1.0 molal: Correctly averaged to 10.1 S/m
  ✅ 2.0 molal: Correctly averaged to 15.2 S/m
  ✅ Uncertainty: Correctly calculated RMS average (0.2121)

======================================================================
Test Summary
======================================================================

✅ PASS: Missing conductivity_Sm column
✅ PASS: All NaN conductivity values
✅ PASS: Partial missing conductivity
✅ PASS: Concentration grouping

----------------------------------------------------------------------
Results: 4/4 tests passed
----------------------------------------------------------------------

🎉 All tests passed! Error handling is working correctly.
```

**What it verifies:**
- ✅ Gracefully handles missing Gamry conductivity data
- ✅ Shows clear, actionable error messages
- ✅ Filters out NaN values from overlays
- ✅ Benchtop plots always generate even if Gamry data incomplete
- ✅ Correctly groups and averages replicates by concentration
- ✅ Calculates proper RMS uncertainties for grouped data

**When to run:**
- After modifying Gamry data loading code
- After changing error handling logic
- After updating grouping/averaging algorithms
- Before committing changes to mahboub2026_plots.py
- As regression test before deployment

---

## Integration Tests

### Test GUI Workflow

**Manual test** - verify GUI works end-to-end:

```bash
# 1. Launch GUI with test data
python gamry_HiPOZ.py --dates 20250813Mahboub2026

# 2. Verify in GUI:
#    - Standards marked with conductivity values
#    - Measurements associated with calibration group
#    - Table shows P, T, composition
#    - Can edit comp and w_ppt columns
#    - Exclude feature works

# 3. Export and check:
#    - Results saved to data/20250813Mahboub2026/
#    - CSV and JSON both updated
#    - Computed conductivities present
```

**Expected behavior:**
- ✅ GUI loads without errors
- ✅ Standards row highlighted (visual indicator)
- ✅ Table editable (comp, w_ppt columns)
- ✅ Calibration column shows associations
- ✅ Can generate plots (Bode, Nyquist, S vs P)
- ✅ Export creates results CSV in same directory

---

### Test Headless Mode

**Automated analysis without GUI:**

```bash
# Run headless analysis
python gamry_HiPOZ.py --headless --dates 20250813Mahboub2026

# Check output
ls -lh data/20250813Mahboub2026/hipoz_*_results.csv
cat data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv | grep conductivity_Sm
```

**Expected output:**
```
[INFO] HEADLESS ANALYSIS MODE
[INFO] Processing Group 1
[INFO]   Standards: 1
[INFO]   Measurements: 12
[INFO]   Cell constant: K = 1.234e-02 ± 5.6e-04 S/m·Ω
[INFO] Results saved to: data/.../hipoz_20260319_143052_results.csv
[INFO] Total measurements analyzed: 12
✓ Original config files updated with computed conductivities
```

**What it verifies:**
- ✅ Headless mode completes without GUI
- ✅ Cell constant computed from standards
- ✅ Conductivities computed for measurements
- ✅ Results CSV created with uncertainties
- ✅ Original config files updated with computed σ

---

### Test Harmonization

**CSV ↔ JSON synchronization:**

```bash
# Test 1: CSV → JSON
python gamry_HiPOZ.py --harmonize data/20250813/zAnalysis20250813.csv

# Verify
ls -lh data/20250813/zAnalysis20250813.json
cat data/20250813/zAnalysis20250813.json | head -20

# Test 2: JSON → CSV (edit JSON first)
python gamry_HiPOZ.py --harmonize data/20250813/zAnalysis20250813.json

# Verify
head -5 data/20250813/zAnalysis20250813.csv
```

**Expected behavior:**
- ✅ CSV → JSON creates/updates JSON file
- ✅ JSON → CSV creates/updates CSV file
- ✅ All fields preserved (notes, comp, P, T)
- ✅ Conductivity values synchronized
- ✅ Exclude flags converted correctly (`x` ↔ `true`)

---

### Test Directory Selection

**GUI dialog and --dates flag:**

```bash
# Test 1: GUI dialog
python gamry_HiPOZ.py
# → Dialog opens, select data/20250813
# → Analysis proceeds

# Test 2: Command line
python gamry_HiPOZ.py --dates 20250813

# Test 3: Multiple directories
python gamry_HiPOZ.py --dates 20250813 20250815

# Test 4: Cancel dialog
python gamry_HiPOZ.py
# → Click Cancel
# → Should exit gracefully
```

**Expected behavior:**
- ✅ Dialog opens in data/ directory
- ✅ Can navigate and select folders
- ✅ --dates flag bypasses dialog
- ✅ Multiple directories processed in sequence
- ✅ Cancel exits without error

---

## Continuous Integration

### Pre-Commit Tests

Run before committing:
```bash
# All automated tests
python test_format_harmonization.py && \
python test_conductivity_writeback.py && \
python test_mahboub_analysis.py

# Quick syntax check
python -m py_compile gamry_HiPOZ.py DataSelector.py gamryTools.py

# Check imports
python -c "import gamryTools; import analysis_config; import harmonize_config"
```

### Regression Tests

**After major changes, verify:**
1. Old config files still load correctly
2. Existing results CSVs unchanged
3. GUI layout still functional
4. Plots still generate
5. Headless mode still works

---

## Test Data

### Required Test Datasets

**Minimal test data** (included):
- `data/20250813/` - Small dataset (1 standard, 12 measurements)
- `data/20250813Mahboub2026/` - Same data with longer directory name

**Extended test data** (optional):
- Multiple calibration groups
- Excluded files
- Various compositions (KCl, NaCl, MgSO4)
- Wide pressure/temperature ranges

### Creating Test Data

To create new test datasets:

```bash
# 1. Create directory structure
mkdir -p data/TestData/ConductivityData_Default

# 2. Copy Gamry output files
cp /path/to/real/data/*.txt data/TestData/ConductivityData_Default/

# 3. Create config file
python analysis_config.py scan data/TestData

# 4. Edit config to mark standards
# (Open CSV in Excel, add conductivity_Sm values)

# 5. Run test
python gamry_HiPOZ.py --dates TestData
```

---

## Troubleshooting Tests

### Test Failures

**"Config file not found"**
- Ensure data directories exist
- Check config file naming: `zAnalysis<date>.csv`

**"Cannot parse S or Z"**
- Verify standards have `conductivity_Sm` values in CSV
- Check that conductivity is numeric (not empty)

**"No valid measurements"**
- Ensure .txt files exist in `ConductivityData_Default/`
- Check file format matches Gamry output

**Import errors**
- Install missing dependencies: `pip install impedance PyQt5`
- Check Python version (3.7+)

### Test Performance

**Slow tests:**
- Circuit fitting is CPU-intensive (expected)
- Use `--dates` to limit scope
- Consider smaller test datasets

**Memory issues:**
- Large datasets (>100 files) may need more RAM
- Process in batches using multiple `--dates` calls

---

## Test Coverage

### Currently Tested

✅ **Config Format Handling**
- CSV loading and saving
- JSON loading and saving
- Format harmonization
- Exclude feature

✅ **Conductivity Write-Back**
- CSV update
- JSON update
- Metadata preservation
- Both formats synchronized

✅ **Analysis Workflow**
- Data loading
- Config detection
- Standard identification
- Measurement processing

### Not Yet Tested (Manual Verification)

⚠️ **GUI Features**
- Button interactions
- Table editing
- Plot generation
- Export functionality

⚠️ **Circuit Fitting**
- CPE model convergence
- RC model convergence
- Uncertainty quantification

⚠️ **Edge Cases**
- Empty config files
- Malformed data files
- Missing pressure/temperature
- Network drive paths

---

## Contributing Tests

When adding new features, please add corresponding tests:

1. **Unit tests** - Test individual functions in isolation
2. **Integration tests** - Test feature end-to-end
3. **Regression tests** - Ensure old behavior unchanged

**Test template:**
```python
#!/usr/bin/env python
"""Test description."""

import sys
sys.path.insert(0, '.')

def test_feature():
    """Test specific feature."""
    print("=== Test: Feature Name ===\n")

    # Setup
    # ...

    # Execute
    # ...

    # Verify
    assert condition, "Error message"

    print("✓ Test passed\n")
    return True

if __name__ == '__main__':
    try:
        test_feature()
        print("✓ ALL TESTS PASSED")
    except AssertionError as e:
        print(f"✗ TEST FAILED: {e}")
        sys.exit(1)
```

---

## See Also

- **[README.md](README.md)** - Project overview and installation
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)** - Bug fixes and validation
