#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script to check that plots match MahboubEtAl2026.py formatting requirements.

Tests:
1. Data points are NOT connected with lines
2. McCleskey model shows as dashed lines
3. Composition limits show as vertical lines
4. LaTeX formatting is correct
5. tab10 colormap is used
"""

import os
import sys
import re

def check_file_content(filepath, checks):
    """
    Check file content against requirements.

    Parameters
    ----------
    filepath : str
        Path to file to check
    checks : list of dict
        Each dict has 'pattern', 'should_exist', 'description'

    Returns
    -------
    passed : bool
        True if all checks pass
    results : list of str
        Check results
    """
    if not os.path.exists(filepath):
        return False, [f"❌ File not found: {filepath}"]

    with open(filepath, 'r') as f:
        content = f.read()

    results = []
    all_passed = True

    for check in checks:
        pattern = check['pattern']
        should_exist = check['should_exist']
        description = check['description']

        found = re.search(pattern, content, re.MULTILINE | re.DOTALL)

        if should_exist:
            if found:
                results.append(f"✓ {description}")
            else:
                results.append(f"✗ {description} (NOT FOUND)")
                all_passed = False
        else:
            if not found:
                results.append(f"✓ {description}")
            else:
                results.append(f"✗ {description} (FOUND - should not exist)")
                all_passed = False

    return all_passed, results


def main():
    print("=" * 70)
    print("Plot Formatting Verification")
    print("=" * 70)
    print()

    all_tests_passed = True

    # ========================================
    # Test 1: plotting.py - No line connections for data
    # ========================================

    print("Test 1: Data points should NOT be connected with lines")
    print("-" * 70)

    checks = [
        {
            'pattern': r"fmt='o'",
            'should_exist': True,
            'description': "Data points use fmt='o' (markers only)"
        },
        {
            'pattern': r"safe_errorbar\(ax1, conc, sigma,.*fmt='o'",
            'should_exist': True,
            'description': "Concentration plot data uses markers only"
        },
        {
            'pattern': r"safe_errorbar\(ax1, temps, sigma,.*fmt='o'",
            'should_exist': True,
            'description': "Temperature plot data uses markers only"
        },
        {
            'pattern': r"# Plot experimental data \(markers only.*\n.*for i.*\n.*safe_errorbar.*fmt='o'",
            'should_exist': True,
            'description': "Experimental data uses fmt='o' markers"
        }
    ]

    passed, results = check_file_content('plotting.py', checks)
    for result in results:
        print(f"  {result}")

    if not passed:
        all_tests_passed = False
    print()

    # ========================================
    # Test 2: plotting.py - McCleskey model as dashed lines
    # ========================================

    print("Test 2: McCleskey model should show as dashed lines")
    print("-" * 70)

    checks = [
        {
            'pattern': r'ax1\.plot\(conc, sigma_model, ls="--"',
            'should_exist': True,
            'description': "Model uses ax1.plot() with dashed lines"
        },
        {
            'pattern': r'ax1\.plot\(temps, sigma_model, ls="--"',
            'should_exist': True,
            'description': "Temperature model uses dashed lines"
        },
        {
            'pattern': r'lw=LineWidth.*alpha=0\.7',
            'should_exist': True,
            'description': "Model lines have proper styling"
        }
    ]

    passed, results = check_file_content('plotting.py', checks)
    for result in results:
        print(f"  {result}")

    if not passed:
        all_tests_passed = False
    print()

    # ========================================
    # Test 3: plotting.py - Composition limit vertical lines
    # ========================================

    print("Test 3: Composition limits should show as vertical lines with labels")
    print("-" * 70)

    checks = [
        {
            'pattern': r'ax1\.axvline\(limit',
            'should_exist': True,
            'description': "Vertical line drawn at composition limit"
        },
        {
            'pattern': r'McCleskey limit',
            'should_exist': True,
            'description': "McCleskey limit text label present"
        },
        {
            'pattern': r'bbox=dict\(boxstyle="round,pad=0\.3"',
            'should_exist': True,
            'description': "Text box styling for limit label"
        }
    ]

    passed, results = check_file_content('plotting.py', checks)
    for result in results:
        print(f"  {result}")

    if not passed:
        all_tests_passed = False
    print()

    # ========================================
    # Test 4: plotting.py - LaTeX formatting
    # ========================================

    print("Test 4: LaTeX formatting should be configured correctly")
    print("-" * 70)

    checks = [
        {
            'pattern': r"plt\.rcParams\['text\.usetex'\] = True",
            'should_exist': True,
            'description': "LaTeX rendering enabled"
        },
        {
            'pattern': r'\\usepackage\{stix\}',
            'should_exist': True,
            'description': "STIX fonts package loaded"
        },
        {
            'pattern': r'\\usepackage\{siunitx\}',
            'should_exist': True,
            'description': "siunitx package loaded"
        },
        {
            'pattern': r'\\usepackage\{upgreek\}',
            'should_exist': True,
            'description': "upgreek package loaded"
        },
        {
            'pattern': r'\\usepackage\[version=4\]\{mhchem\}',
            'should_exist': True,
            'description': "mhchem package loaded"
        }
    ]

    passed, results = check_file_content('plotting.py', checks)
    for result in results:
        print(f"  {result}")

    if not passed:
        all_tests_passed = False
    print()

    # ========================================
    # Test 5: plot_cortes_publication.py - No line connections
    # ========================================

    print("Test 5: Cortes plots should NOT connect data points")
    print("-" * 70)

    checks = [
        {
            'pattern': r"fmt='o-'",
            'should_exist': False,
            'description': "No 'o-' format (would create lines)"
        },
        {
            'pattern': r"fmt='o'",
            'should_exist': True,
            'description': "Uses 'o' format (markers only)"
        },
        {
            'pattern': r"mew=1\.5",
            'should_exist': True,
            'description': "Marker edge width specified"
        }
    ]

    passed, results = check_file_content('plot_cortes_publication.py', checks)
    for result in results:
        print(f"  {result}")

    if not passed:
        all_tests_passed = False
    print()

    # ========================================
    # Test 6: mahboub2026_plots.py - McCleskey limits defined
    # ========================================

    print("Test 6: Mahboub plots should use McCleskey limits")
    print("-" * 70)

    checks = [
        {
            'pattern': r'MCCLESKEY_LIMITS = \{',
            'should_exist': True,
            'description': "McCleskey limits dictionary defined"
        },
        {
            'pattern': r"'NaCl': 0\.9999",
            'should_exist': True,
            'description': "NaCl limit defined"
        },
        {
            'pattern': r"'MgSO4': 0\.01245",
            'should_exist': True,
            'description': "MgSO4 limit defined"
        },
        {
            'pattern': r"mccleskey_limit=MCCLESKEY_LIMITS",
            'should_exist': True,
            'description': "Limits passed to plotting functions"
        }
    ]

    passed, results = check_file_content('mahboub2026/mahboub2026_plots.py', checks)
    for result in results:
        print(f"  {result}")

    if not passed:
        all_tests_passed = False
    print()

    # ========================================
    # Test 7: colormap usage
    # ========================================

    print("Test 7: Colormap should be tab10 (Cortes preference)")
    print("-" * 70)

    # Check config_plots.py for colormap settings
    checks_config = [
        {
            'pattern': r"COLORMAP_CONCENTRATION = 'tab10'",
            'should_exist': True,
            'description': "Concentration colormap set to tab10"
        },
        {
            'pattern': r"COLORMAP_TEMPERATURE = 'tab10'",
            'should_exist': True,
            'description': "Temperature colormap set to tab10"
        }
    ]

    passed, results = check_file_content('config_plots.py', checks_config)
    print(f"  config_plots.py (centralized settings):")
    for result in results:
        print(f"    {result}")
    if not passed:
        all_tests_passed = False

    # Check that study files import from config_plots
    checks_import = [
        {
            'pattern': r"from config_plots import.*COLORMAP",
            'should_exist': True,
            'description': "Imports colormap from config_plots"
        }
    ]

    for filepath in ['mahboub2026/mahboub2026_plots.py']:
        passed, results = check_file_content(filepath, checks_import)
        print(f"  {os.path.basename(filepath)}:")
        for result in results:
            print(f"    {result}")
        if not passed:
            all_tests_passed = False
    print()

    # ========================================
    # Summary
    # ========================================

    print("=" * 70)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("Plot formatting matches MahboubEtAl2026.py requirements:")
        print("  • Data points: markers only (no line connections)")
        print("  • McCleskey model: dashed lines")
        print("  • Composition limits: vertical lines with labels")
        print("  • LaTeX formatting: properly configured")
        print("  • Colormap: tab10 (Cortes color scheme)")
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("Please review the failures above and make necessary corrections.")
        return 1

    print("=" * 70)
    print()

    print("Next steps:")
    print("  1. Generate plots: python3 mahboub2026/mahboub2026_plots.py")
    print("  2. Generate Cortes plots: python3 plot_cortes_publication.py")
    print("  3. Visually inspect PDFs to confirm formatting")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
