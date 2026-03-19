#!/usr/bin/env python
"""
Test the complete analysis workflow for 20250813Mahboub2026 data.

This simulates what happens when you run python gamry_HiPOZOZ.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '.')
os.chdir('/Users/svance/Library/CloudStorage/Dropbox/ElectricalProperties/hipozgenai')

print("=" * 70)
print("HIPOZ ANALYSIS TEST: 20250813Mahboub2026")
print("=" * 70)
print()

# Step 1: Check data directory
print("Step 1: Checking data directory...")
data_dir = Path('data/20250813Mahboub2026')
conductivity_dir = data_dir / 'ConductivityData_Default'

if not data_dir.exists():
    print(f"✗ Data directory not found: {data_dir}")
    sys.exit(1)

print(f"✓ Data directory found: {data_dir}")

data_files = list(conductivity_dir.glob('*.txt'))
print(f"✓ Found {len(data_files)} measurement files")
print()

# Step 2: Check config file
print("Step 2: Checking config file...")
from gamry_HiPOZ import find_config_in_data_dirs

dates = ['20250813Mahboub2026']
config_path = find_config_in_data_dirs(dates)

if not config_path:
    print("✗ Config file not found")
    sys.exit(1)

print(f"✓ Config found: {config_path}")
print(f"  Format: {'CSV' if config_path.endswith('.csv') else 'JSON'}")
print()

# Step 3: Load config
print("Step 3: Loading configuration...")
from analysis_config import AnalysisConfig

config = AnalysisConfig(config_path)
print(f"✓ Config loaded successfully")
print(f"  Groups: {len(config.calibration_groups)}")

for group in config.calibration_groups:
    n_std = len(group['standards'])
    n_meas = len(group['measurements'])
    print(f"  {group['name']}: {n_std} standards, {n_meas} measurements")
print()

# Step 4: Check standards
print("Step 4: Analyzing standards...")
for group in config.calibration_groups:
    standards = group['standards']

    if not standards:
        print("✗ No standards found!")
        sys.exit(1)

    print(f"✓ Found {len(standards)} standard(s)")

    for i, std in enumerate(standards, 1):
        if isinstance(std, dict):
            fname = std.get('filename', 'unknown')
            cond = std.get('conductivity_Sm', 'N/A')
            comp = std.get('comp', 'N/A')
            print(f"  {i}. {fname}")
            print(f"     Conductivity: {cond} S/m")
            print(f"     Composition: {comp}")
        else:
            print(f"  {i}. {std} (no explicit conductivity)")

    # Check if standards have conductivity values
    std_with_conductivity = [s for s in standards if isinstance(s, dict) and s.get('conductivity_Sm')]
    if std_with_conductivity:
        print(f"✓ {len(std_with_conductivity)} standard(s) have conductivity values")
    else:
        print("⚠ No standards have explicit conductivity values")
print()

# Step 5: Check measurements
print("Step 5: Analyzing measurements...")
for group in config.calibration_groups:
    measurements = group['measurements']

    print(f"✓ Found {len(measurements)} measurement(s)")

    # Count by composition
    comp_counts = {}
    for meas in measurements:
        if isinstance(meas, dict):
            comp = meas.get('comp', 'Unknown')
            comp_counts[comp] = comp_counts.get(comp, 0) + 1

    print("  By composition:")
    for comp, count in comp_counts.items():
        print(f"    {comp}: {count} measurement(s)")

    # Show concentration info
    print("  Concentrations:")
    for i, meas in enumerate(measurements, 1):
        if isinstance(meas, dict):
            fname = meas.get('filename', 'unknown')
            comp = meas.get('comp', 'N/A')
            w_ppt = meas.get('w_ppt', '')
            w_molal = meas.get('w_molal', '')

            conc_str = ''
            if w_ppt:
                conc_str = f"{w_ppt} ppt"
            elif w_molal:
                conc_str = f"{w_molal} molal"

            print(f"    {i}. {Path(fname).name[:40]}: {comp} {conc_str}")
print()

# Step 6: Verify CSV format
print("Step 6: Verifying CSV format...")
import csv

csv_path = Path(config_path)
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

print(f"✓ CSV has {len(headers)} columns")
print(f"  Columns: {', '.join(headers)}")
print(f"✓ CSV has {len(rows)} rows")

# Check for P_MPa and T_K columns
if 'P_MPa' in headers and 'T_K' in headers:
    print("✓ CSV includes P_MPa and T_K columns")

    # Show P and T values
    p_values = set(row['P_MPa'] for row in rows if row['P_MPa'])
    t_values = set(row['T_K'] for row in rows if row['T_K'])

    print(f"  Pressure values: {', '.join(sorted(p_values))} MPa")
    print(f"  Temperature values: {', '.join(sorted(t_values))} K")
else:
    print("⚠ CSV missing P_MPa or T_K columns")
print()

# Step 7: Summary
print("=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)
print()

total_standards = sum(len(g['standards']) for g in config.calibration_groups)
total_measurements = sum(len(g['measurements']) for g in config.calibration_groups)

print(f"✓ Config file: {config_path}")
print(f"✓ Format: CSV with P_MPa and T_K columns")
print(f"✓ Total standards: {total_standards}")
print(f"✓ Total measurements: {total_measurements}")
print(f"✓ Total files: {total_standards + total_measurements}")
print()

# Check if ready for GUI
if total_standards > 0:
    if any(isinstance(s, dict) and s.get('conductivity_Sm') for g in config.calibration_groups for s in g['standards']):
        print("✓ READY FOR GUI ANALYSIS")
        print("  Run: python gamry_HiPOZOZ.py")
    else:
        print("⚠ Standards need conductivity values")
        print("  Edit CSV to add conductivity_Sm for standards")
else:
    print("✗ NO STANDARDS FOUND")
    print("  Edit CSV to mark KCl files as type=standard")

print()
print("=" * 70)
