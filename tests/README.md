# HiPOZ Test Suite

Automated tests for HiPOZ impedance analysis system.

## Quick Test

Run all tests:
```bash
# From repo root
python tests/test_format_harmonization.py
python tests/test_conductivity_writeback.py
python tests/test_mahboub_analysis.py
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

### Not Yet Tested ⚠️
- GUI interactions
- Circuit fitting convergence
- Plot generation (visual verification)
- Edge cases (empty configs, malformed files)

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
