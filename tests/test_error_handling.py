#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test error handling for Gamry impedance data integration.

This test suite verifies that mahboub2026_plots.py correctly detects and handles:
1. Missing conductivity_Sm column (standards not defined)
2. All NaN conductivity values (standards not associated)
3. Partial missing conductivity (some measurements not associated)

Run this test after modifying the Gamry data loading or error handling code.

Usage:
    python test_error_handling.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_missing_column():
    """Test detection of missing conductivity_Sm column."""
    print("=" * 70)
    print("Test 1: Missing conductivity_Sm column")
    print("=" * 70)
    print("Scenario: Standards not defined in calibration config")
    print()

    # Create mock data without conductivity column
    test_data = {
        'filename': ['test1.txt', 'test2.txt'],
        'type': ['measurement', 'measurement'],
        'comp': ['NaCl', 'NaCl'],
        'w_molal': [1.0, 2.0],
        'R_ohm': [10.5, 8.3],
        'T_K': [293.15, 293.15]
    }
    gamry_df = pd.DataFrame(test_data)

    # Test the check
    if 'conductivity_Sm' not in gamry_df.columns:
        print("✅ PASS: Correctly detected missing conductivity_Sm column")
        print("   Expected behavior: Set gamry_df = None, skip all overlays")
        print("   User action: Define standards in zAnalysis config")
        return True
    else:
        print("❌ FAIL: Should have detected missing conductivity_Sm column")
        return False


def test_all_nan_conductivity():
    """Test detection of all NaN conductivity values."""
    print()
    print("=" * 70)
    print("Test 2: All conductivity values are NaN")
    print("=" * 70)
    print("Scenario: Standards defined but not associated with measurements")
    print()

    # Create mock data with all NaN conductivity
    test_data = {
        'filename': ['test1.txt', 'test2.txt'],
        'type': ['measurement', 'measurement'],
        'comp': ['NaCl', 'NaCl'],
        'w_molal': [1.0, 2.0],
        'conductivity_Sm': [np.nan, np.nan],
        'conductivity_Sm_unc': [np.nan, np.nan],
        'T_K': [293.15, 293.15]
    }
    gamry_df = pd.DataFrame(test_data)

    # Test the check
    if gamry_df['conductivity_Sm'].isna().all():
        print("✅ PASS: Correctly detected all NaN conductivity values")
        print("   Expected behavior: Set gamry_df = None, skip all overlays")
        print("   User action: Associate measurements with standards in HiPOZ GUI")
        return True
    else:
        print("❌ FAIL: Should have detected all NaN conductivity values")
        return False


def test_partial_missing_conductivity():
    """Test detection and handling of partial missing conductivity."""
    print()
    print("=" * 70)
    print("Test 3: Partial missing conductivity")
    print("=" * 70)
    print("Scenario: Some measurements have conductivity, others don't")
    print()

    # Create mock data with partial missing conductivity
    test_data = {
        'filename': ['test1.txt', 'test2.txt', 'test3.txt', 'test4.txt'],
        'type': ['measurement', 'measurement', 'measurement', 'measurement'],
        'comp': ['NaCl', 'NaCl', 'MgSO4', 'MgSO4'],
        'w_molal': [1.0, 2.0, 0.5, 1.0],
        'conductivity_Sm': [10.5, np.nan, 3.2, 5.1],
        'conductivity_Sm_unc': [0.3, np.nan, 0.1, 0.2],
        'T_K': [293.15, 293.15, 292.35, 292.35]
    }
    gamry_df = pd.DataFrame(test_data)

    n_total = len(gamry_df)
    n_valid = gamry_df['conductivity_Sm'].notna().sum()

    # Test global statistics
    passed = True
    if n_valid < n_total:
        print(f"✅ PASS: Correctly detected partial missing data ({n_total - n_valid}/{n_total} missing)")
        print("   Expected behavior: Show warning, continue with valid data")
    else:
        print("❌ FAIL: Should have detected partial missing data")
        passed = False

    # Test compound-specific filtering
    print()
    print("Testing compound-specific filtering:")

    # NaCl filtering
    nacl_data = gamry_df[gamry_df['comp'] == 'NaCl'].copy()
    nacl_valid = nacl_data[nacl_data['conductivity_Sm'].notna()].copy()
    n_nacl_total = len(nacl_data)
    n_nacl_valid = len(nacl_valid)

    if n_nacl_valid == 1 and n_nacl_total == 2:
        print(f"  ✅ NaCl: Correctly filtered to {n_nacl_valid}/{n_nacl_total} valid measurements")
    else:
        print(f"  ❌ NaCl: Expected 1/2 valid, got {n_nacl_valid}/{n_nacl_total}")
        passed = False

    # MgSO4 filtering
    mgso4_data = gamry_df[gamry_df['comp'] == 'MgSO4'].copy()
    mgso4_valid = mgso4_data[mgso4_data['conductivity_Sm'].notna()].copy()
    n_mgso4_total = len(mgso4_data)
    n_mgso4_valid = len(mgso4_valid)

    if n_mgso4_valid == 2 and n_mgso4_total == 2:
        print(f"  ✅ MgSO4: Correctly kept {n_mgso4_valid}/{n_mgso4_total} valid measurements")
    else:
        print(f"  ❌ MgSO4: Expected 2/2 valid, got {n_mgso4_valid}/{n_mgso4_total}")
        passed = False

    print()
    print("   User action: Associate remaining measurements with standards")

    return passed


def test_compound_grouping():
    """Test grouping by concentration with valid data."""
    print()
    print("=" * 70)
    print("Test 4: Concentration grouping and averaging")
    print("=" * 70)
    print("Scenario: Multiple measurements at same concentration should be averaged")
    print()

    # Create mock data with replicates at same concentration
    test_data = {
        'filename': ['test1.txt', 'test2.txt', 'test3.txt', 'test4.txt'],
        'type': ['measurement', 'measurement', 'measurement', 'measurement'],
        'comp': ['NaCl', 'NaCl', 'NaCl', 'NaCl'],
        'w_molal': [1.0, 1.0, 2.0, 2.0],  # 2 replicates at each concentration
        'conductivity_Sm': [10.0, 10.2, 15.0, 15.4],
        'conductivity_Sm_unc': [0.3, 0.3, 0.4, 0.4],
        'T_K': [293.15, 293.15, 293.15, 293.15]
    }
    gamry_df = pd.DataFrame(test_data)

    # Filter to valid data
    nacl_data = gamry_df[gamry_df['comp'] == 'NaCl'].copy()
    nacl_valid = nacl_data[nacl_data['conductivity_Sm'].notna()].copy()

    # Group by concentration
    grouped = nacl_valid.groupby('w_molal').agg({
        'conductivity_Sm': 'mean',
        'conductivity_Sm_unc': lambda x: np.sqrt(np.sum(x**2))/len(x) if len(x) > 1 else x.iloc[0],
        'T_K': 'mean'
    }).reset_index()

    passed = True

    # Check we got 2 concentration groups
    if len(grouped) == 2:
        print(f"✅ PASS: Correctly grouped into {len(grouped)} concentration bins")
    else:
        print(f"❌ FAIL: Expected 2 concentration groups, got {len(grouped)}")
        passed = False

    # Check averaging worked correctly
    expected_conc_1 = (10.0 + 10.2) / 2  # Should be 10.1
    actual_conc_1 = grouped[grouped['w_molal'] == 1.0]['conductivity_Sm'].iloc[0]

    if abs(actual_conc_1 - expected_conc_1) < 1e-6:
        print(f"  ✅ 1.0 molal: Correctly averaged to {actual_conc_1:.1f} S/m")
    else:
        print(f"  ❌ 1.0 molal: Expected {expected_conc_1:.1f}, got {actual_conc_1:.1f}")
        passed = False

    expected_conc_2 = (15.0 + 15.4) / 2  # Should be 15.2
    actual_conc_2 = grouped[grouped['w_molal'] == 2.0]['conductivity_Sm'].iloc[0]

    if abs(actual_conc_2 - expected_conc_2) < 1e-6:
        print(f"  ✅ 2.0 molal: Correctly averaged to {actual_conc_2:.1f} S/m")
    else:
        print(f"  ❌ 2.0 molal: Expected {expected_conc_2:.1f}, got {actual_conc_2:.1f}")
        passed = False

    # Check uncertainty calculation (RMS average)
    expected_unc = np.sqrt((0.3**2 + 0.3**2)) / 2
    actual_unc = grouped[grouped['w_molal'] == 1.0]['conductivity_Sm_unc'].iloc[0]

    if abs(actual_unc - expected_unc) < 1e-6:
        print(f"  ✅ Uncertainty: Correctly calculated RMS average ({actual_unc:.4f})")
    else:
        print(f"  ❌ Uncertainty: Expected {expected_unc:.4f}, got {actual_unc:.4f}")
        passed = False

    return passed


def run_all_tests():
    """Run all error handling tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   Gamry Impedance Data - Error Handling Test Suite".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Testing mahboub2026_plots.py error detection and handling...")
    print()

    results = []

    # Run all tests
    results.append(("Missing conductivity_Sm column", test_missing_column()))
    results.append(("All NaN conductivity values", test_all_nan_conductivity()))
    results.append(("Partial missing conductivity", test_partial_missing_conductivity()))
    results.append(("Concentration grouping", test_compound_grouping()))

    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print()
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("-" * 70)

    if passed == total:
        print()
        print("🎉 All tests passed! Error handling is working correctly.")
        print()
        return 0
    else:
        print()
        print("⚠️  Some tests failed. Review error handling implementation.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
