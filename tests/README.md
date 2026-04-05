# HiPOZ Test Suite

Automated tests for HiPOZ impedance analysis system.

## Quick Test

Run all tests:
```bash
# From repo root
python tests/test_format_harmonization.py
python tests/test_conductivity_writeback.py
python tests/test_mahboub_analysis.py
python tests/test_gui_initialization.py
python tests/test_plot_generation.py
python tests/test_latex_tables.py
python tests/test_benchtop_data.py
python mahboub2026/tests/test_error_handling.py
```

All tests should pass with `✓ ALL TESTS PASSED` or `🎉 All tests passed!`

## Test Files

### test_format_harmonization.py
Tests CSV/JSON config file handling.

**What it verifies:**
- ✅ CSV configs remain CSV after GUI saves
- ✅ JSON configs remain JSON after GUI saves
- ✅ Exclude feature works correctly
- ✅ Both formats can coexist

**When to run:** After modifying config I/O or format handling.

### test_conductivity_writeback.py
Tests conductivity calculation write-back to config files.

**What it verifies:**
- ✅ Computed conductivities written to CSV
- ✅ Computed conductivities written to JSON
- ✅ Metadata preserved (P, T, comp, notes)
- ✅ Both formats updated simultaneously

**When to run:** After modifying calibration workflow or conductivity calculations.

### test_mahboub_analysis.py
Integration test for Mahboub 2026 data.

**What it verifies:**
- ✅ Config file auto-detection
- ✅ Standards have conductivity values
- ✅ Measurements have composition metadata
- ✅ P_MPa and T_K columns present

**When to run:** After modifying data loading or Mahboub analysis.

### test_gui_initialization.py
Tests GUI initialization and widget creation.

**What it verifies:**
- ✅ DataSelector GUI initializes without errors
- ✅ All tabs created (Timeseries, Bode & Nyquist, S vs P)
- ✅ Data table populated with correct columns
- ✅ All control buttons present
- ✅ Plot canvases created for each tab
- ✅ DataFrame structure and masks initialized

**When to run:** After modifying GUI structure or widget creation.

### test_plot_generation.py
Tests plot generation for all visualization types.

**What it verifies:**
- ✅ Timeseries plots generate correctly
- ✅ Bode plots (magnitude and phase)
- ✅ Nyquist plots with equal aspect ratio
- ✅ S vs P scatter plots with temperature coloring
- ✅ Plot export to PNG and PDF
- ✅ Error bar plotting
- ✅ Multi-panel figure layout

**When to run:** After modifying plotting functions or visualization features.

### test_latex_tables.py
Tests LaTeX table generation for publications.

**What it verifies:**
- ✅ Basic LaTeX table structure
- ✅ Uncertainty formatting (value ± error)
- ✅ Special characters (subscripts, superscripts)
- ✅ Scientific notation
- ✅ Multi-row headers
- ✅ Column alignment (left, center, right)
- ✅ Booktabs professional styling
- ✅ Export to .tex files

**When to run:** After modifying data export or table formatting.

### test_benchtop_data.py
Tests benchtop data accuracy against reference values.

**What it verifies:**
- ✅ JesusData2025.csv loads correctly
- ✅ NaCl:MgSO4 data extraction
- ✅ KCl data extraction
- ✅ Unit conversions (mS/cm ↔ S/m)
- ✅ Temperature dependence (S increases with T)
- ✅ Concentration dependence (S increases with conc)
- ✅ Replicate measurement consistency

**When to run:** After processing new Cortes benchtop measurements.

## Study-Specific Tests

### mahboub2026/tests/test_error_handling.py
Tests Gamry impedance data error handling.

**What it verifies:**
- ✅ Missing conductivity column detection
- ✅ All NaN conductivity detection
- ✅ Partial missing data handling
- ✅ Concentration grouping and averaging

**When to run:** After modifying Gamry integration or error handling.

## Running Tests

### Individual Test
```bash
python tests/test_format_harmonization.py
```

### All General Tests
```bash
for test in tests/test_*.py; do python "$test"; done
```

### All Tests (including study-specific)
```bash
for test in tests/test_*.py mahboub2026/tests/test_*.py; do
    echo "Running $test..."
    python "$test" || exit 1
done
echo "✓ All tests passed!"
```

## Writing New Tests

### Test Template
```python
#!/usr/bin/env python
"""Test description."""

import sys
sys.path.insert(0, '.')

def test_feature():
    """Test specific feature."""
    print("=== Test: Feature Name ===\n")

    # Setup test data
    test_data = {...}

    # Execute code being tested
    result = function_to_test(test_data)

    # Verify expected behavior
    assert result == expected, f"Expected {expected}, got {result}"

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

### Test Guidelines
- **One file per feature area** (format handling, calibration, plotting)
- **Clear test names** that describe what's being tested
- **Print progress** so failures are easy to diagnose
- **Assert with messages** to explain failures
- **Exit 1 on failure** for CI/CD integration

## Pre-Commit Checklist

Before committing code changes:
```bash
# 1. Run all tests
for test in tests/test_*.py mahboub2026/tests/test_*.py; do
    python "$test" || exit 1
done

# 2. Check syntax
python -m py_compile gamry_HiP.py gamryTools.py

# 3. Verify imports
python -c "import gamryTools; import gamry_integration"
```

## Test Coverage

### Currently Tested ✅
- Config format handling (CSV/JSON)
- Conductivity write-back
- Mahboub analysis workflow
- Gamry error handling
- Data grouping and averaging
- GUI initialization and widget creation
- Plot generation (all types: timeseries, Bode, Nyquist, S vs P)
- LaTeX table generation for publications
- Benchtop data accuracy vs reference values

### Not Yet Tested ⚠️
- Interactive GUI operations (button clicks, selections)
- Circuit fitting convergence edge cases
- Real-time plot updates during data selection
- Edge cases (empty configs, malformed files)
- Network-dependent operations

## Troubleshooting

### "Module not found" errors
```bash
# Ensure tests run from repo root
cd /path/to/hipozgenai
python tests/test_name.py
```

### "Config file not found"
```bash
# Check test data exists
ls -lh data/20250813/zAnalysis20250813.csv
```

### Tests pass locally but fail in CI
- Check Python version (3.7+)
- Verify all dependencies installed
- Ensure data files committed to repo

## See Also

- [Testing Guide](../docs/TESTING.md) - Detailed testing documentation
- [Main README](../README.md) - Project overview
- [Contributing](#) - How to contribute tests
