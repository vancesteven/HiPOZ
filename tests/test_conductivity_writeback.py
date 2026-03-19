#!/usr/bin/env python
"""
Test conductivity write-back feature.

Verifies that computed conductivity values from headless analysis
are written back to the original config files (both CSV and JSON).
"""

import sys
import os
from pathlib import Path
import csv
import json
import shutil
import tempfile

# Set up paths
sys.path.insert(0, '.')
os.chdir('/Users/svance/Library/CloudStorage/Dropbox/ElectricalProperties/hipozgenai')

from headless_analysis import _update_config_with_conductivities
from analysis_config import AnalysisConfig


def test_writeback_to_csv():
    """Test writing computed conductivities back to CSV config."""
    print("=== Test 1: Write-Back to CSV Config ===\n")

    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test CSV config
        csv_path = temp_path / 'test_config.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'group_name', 'filename', 'P_MPa', 'T_K', 'type',
                'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes'
            ])
            writer.writeheader()

            # Standard with known conductivity
            writer.writerow({
                'group_name': 'Test Group',
                'filename': 'standard_001.txt',
                'P_MPa': 0.1,
                'T_K': 292,
                'type': 'standard',
                'conductivity_Sm': 0.0084,
                'comp': 'KCl',
                'w_ppt': 1000,
                'w_molal': '',
                'exclude': '',
                'notes': 'Test standard'
            })

            # Measurements without conductivity (to be computed)
            for i in range(1, 4):
                writer.writerow({
                    'group_name': 'Test Group',
                    'filename': f'measurement_{i:03d}.txt',
                    'P_MPa': 10 * i,
                    'T_K': 292,
                    'type': 'measurement',
                    'conductivity_Sm': '',  # Empty - will be filled
                    'comp': 'NaCl',
                    'w_ppt': 1000,
                    'w_molal': '',
                    'exclude': '',
                    'notes': f'Test measurement {i}'
                })

        # Load config
        config = AnalysisConfig(str(csv_path))

        # Simulate analysis results
        results = [
            {
                'type': 'measurement',
                'filename': 'measurement_001.txt',
                'conductivity_Sm': 0.125,
                'group_name': 'Test Group'
            },
            {
                'type': 'measurement',
                'filename': 'measurement_002.txt',
                'conductivity_Sm': 0.098,
                'group_name': 'Test Group'
            },
            {
                'type': 'measurement',
                'filename': 'measurement_003.txt',
                'conductivity_Sm': 0.073,
                'group_name': 'Test Group'
            }
        ]

        # Write back conductivities
        _update_config_with_conductivities(config, results, temp_path)

        # Verify CSV was updated
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"✓ CSV file updated: {csv_path.name}")
        print(f"  Total rows: {len(rows)}")

        # Check that measurements now have conductivity values
        measurements = [r for r in rows if r['type'] == 'measurement']
        assert len(measurements) == 3, f"Expected 3 measurements, got {len(measurements)}"

        for meas in measurements:
            cond = meas['conductivity_Sm']
            assert cond, f"Measurement {meas['filename']} missing conductivity!"
            assert float(cond) > 0, f"Conductivity should be positive, got {cond}"
            print(f"  ✓ {meas['filename']}: σ = {cond} S/m")

        # Verify notes were preserved
        for meas in measurements:
            assert meas['notes'].startswith('Test measurement'), \
                f"Notes not preserved for {meas['filename']}"

        print("\n✓ CSV write-back verified: Conductivities added, metadata preserved\n")
        return True


def test_writeback_to_json():
    """Test writing computed conductivities back to JSON config."""
    print("=== Test 2: Write-Back to JSON Config ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test JSON config
        json_path = temp_path / 'test_config.json'
        config_data = {
            'calibration_groups': [
                {
                    'name': 'Test Group',
                    'standards': [
                        {
                            'filename': 'standard_001.txt',
                            'conductivity_Sm': 0.0084,
                            'comp': 'KCl',
                            'notes': 'Test standard'
                        }
                    ],
                    'measurements': [
                        {
                            'filename': 'measurement_001.txt',
                            'comp': 'NaCl',
                            'notes': 'Test measurement 1'
                        },
                        {
                            'filename': 'measurement_002.txt',
                            'comp': 'NaCl',
                            'notes': 'Test measurement 2'
                        }
                    ]
                }
            ]
        }

        with open(json_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Load config
        config = AnalysisConfig(str(json_path))

        # Simulate analysis results
        results = [
            {
                'type': 'measurement',
                'filename': 'measurement_001.txt',
                'conductivity_Sm': 0.145
            },
            {
                'type': 'measurement',
                'filename': 'measurement_002.txt',
                'conductivity_Sm': 0.132
            }
        ]

        # Write back conductivities
        _update_config_with_conductivities(config, results, temp_path)

        # Verify JSON was updated
        with open(json_path, 'r') as f:
            updated_data = json.load(f)

        print(f"✓ JSON file updated: {json_path.name}")

        # Check measurements have conductivity values
        group = updated_data['calibration_groups'][0]
        measurements = group['measurements']

        assert len(measurements) == 2, f"Expected 2 measurements, got {len(measurements)}"

        for meas in measurements:
            cond = meas.get('conductivity_Sm')
            assert cond, f"Measurement {meas['filename']} missing conductivity!"
            assert cond > 0, f"Conductivity should be positive, got {cond}"
            print(f"  ✓ {meas['filename']}: σ = {cond} S/m")

        # Verify notes were preserved
        for meas in measurements:
            assert 'Test measurement' in meas.get('notes', ''), \
                f"Notes not preserved for {meas['filename']}"

        print("\n✓ JSON write-back verified: Conductivities added, metadata preserved\n")
        return True


def test_both_formats_updated():
    """Test that both CSV and JSON are updated simultaneously."""
    print("=== Test 3: Both Formats Updated ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create CSV config
        csv_path = temp_path / 'test_config.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'group_name', 'filename', 'P_MPa', 'T_K', 'type',
                'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes'
            ])
            writer.writeheader()
            writer.writerow({
                'group_name': 'Test Group',
                'filename': 'standard_001.txt',
                'type': 'standard',
                'conductivity_Sm': 0.0084,
                'comp': 'KCl',
                'P_MPa': 0.1,
                'T_K': 292,
                'w_ppt': '',
                'w_molal': '',
                'exclude': '',
                'notes': ''
            })
            writer.writerow({
                'group_name': 'Test Group',
                'filename': 'measurement_001.txt',
                'type': 'measurement',
                'conductivity_Sm': '',
                'comp': 'NaCl',
                'P_MPa': 10,
                'T_K': 292,
                'w_ppt': '',
                'w_molal': '',
                'exclude': '',
                'notes': ''
            })

        # Load config (from CSV)
        config = AnalysisConfig(str(csv_path))

        # Simulate analysis results
        results = [
            {
                'type': 'measurement',
                'filename': 'measurement_001.txt',
                'conductivity_Sm': 0.156
            }
        ]

        # Write back (should update both CSV and JSON)
        _update_config_with_conductivities(config, results, temp_path)

        # Check both files exist
        json_path = csv_path.with_suffix('.json')

        assert csv_path.exists(), "CSV file should exist"
        assert json_path.exists(), "JSON file should exist"

        print(f"✓ Both files created:")
        print(f"  - {csv_path.name}")
        print(f"  - {json_path.name}")

        # Verify CSV has conductivity
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            meas_rows = [r for r in rows if r['type'] == 'measurement']
            assert meas_rows[0]['conductivity_Sm'], "CSV missing conductivity"
            csv_cond = float(meas_rows[0]['conductivity_Sm'])

        # Verify JSON has conductivity
        with open(json_path, 'r') as f:
            data = json.load(f)
            measurements = data['calibration_groups'][0]['measurements']
            assert measurements[0].get('conductivity_Sm'), "JSON missing conductivity"
            json_cond = measurements[0]['conductivity_Sm']

        # Verify values match
        assert abs(csv_cond - json_cond) < 1e-6, \
            f"CSV and JSON conductivities don't match: {csv_cond} vs {json_cond}"

        print(f"✓ Conductivity values match in both formats: {csv_cond} S/m")
        print("\n✓ Both formats synchronized correctly\n")
        return True


def test_preserves_existing_data():
    """Test that existing metadata is preserved during write-back."""
    print("=== Test 4: Preserve Existing Metadata ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create CSV with rich metadata
        csv_path = temp_path / 'test_config.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'group_name', 'filename', 'P_MPa', 'T_K', 'type',
                'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes'
            ])
            writer.writeheader()
            writer.writerow({
                'group_name': 'Test Group',
                'filename': 'standard_001.txt',
                'type': 'standard',
                'conductivity_Sm': 0.0084,
                'comp': 'KCl',
                'P_MPa': 0.1,
                'T_K': 292,
                'w_ppt': 1000,
                'w_molal': 0.0135,
                'exclude': '',
                'notes': 'Bottle #47, calibrated 2025-08-13'
            })
            writer.writerow({
                'group_name': 'Test Group',
                'filename': 'measurement_001.txt',
                'type': 'measurement',
                'conductivity_Sm': '',
                'comp': 'NaCl',
                'P_MPa': 25.5,
                'T_K': 295.3,
                'w_ppt': 35000,
                'w_molal': 0.605,
                'exclude': '',
                'notes': 'Seawater analog, took 3 tries to stabilize'
            })

        # Load and update
        config = AnalysisConfig(str(csv_path))
        results = [
            {
                'type': 'measurement',
                'filename': 'measurement_001.txt',
                'conductivity_Sm': 4.567
            }
        ]

        _update_config_with_conductivities(config, results, temp_path)

        # Verify all metadata preserved
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        meas = [r for r in rows if r['type'] == 'measurement'][0]

        # Define expected values (numeric fields may be float or string)
        checks = {
            'comp': 'NaCl',
            'notes': 'Seawater analog, took 3 tries to stabilize'
        }

        numeric_checks = {
            'P_MPa': 25.5,
            'T_K': 295.3,
            'w_ppt': 35000,
            'w_molal': 0.605
        }

        print("✓ Verifying preserved metadata:")

        # Check string fields
        for field, expected in checks.items():
            actual = meas[field]
            assert actual == expected, f"{field} not preserved: {actual} != {expected}"
            print(f"  ✓ {field}: {actual}")

        # Check numeric fields (may be stored as float or string, compare numerically)
        for field, expected in numeric_checks.items():
            actual = meas[field]
            # Convert to float for comparison
            actual_num = float(actual) if actual else 0
            assert abs(actual_num - expected) < 1e-6, f"{field} not preserved: {actual_num} != {expected}"
            print(f"  ✓ {field}: {actual}")

        # Verify conductivity was added
        assert meas['conductivity_Sm'], "Conductivity not added"
        print(f"  ✓ conductivity_Sm: {meas['conductivity_Sm']} (newly added)")

        print("\n✓ All metadata preserved during write-back\n")
        return True


if __name__ == '__main__':
    print("=" * 70)
    print("CONDUCTIVITY WRITE-BACK TEST SUITE")
    print("=" * 70)
    print()

    try:
        test_writeback_to_csv()
        test_writeback_to_json()
        test_both_formats_updated()
        test_preserves_existing_data()

        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • Computed conductivities written to CSV config")
        print("  • Computed conductivities written to JSON config")
        print("  • Both formats updated simultaneously")
        print("  • All existing metadata preserved (P, T, comp, notes, etc.)")
        print("  • Ready for production use!")
        print()

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
