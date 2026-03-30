#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse JesusData2025.csv from spreadsheet format to standardized long format.

Input: JesusData2025.csv (wide format, multiple compound blocks)
Output: Cortes2026BenchtopData.csv (long format, standardized columns)

Format matches Mahboub2026BenchtopData.csv:
    compound, concentration_molal, temperature_C, temperature_K,
    conductivity_Sm, replicate, source, notes
"""

import pandas as pd
import numpy as np
import re

def parse_temperature(temp_str):
    """Extract temperature in Celsius from string like '20C' or '5C'."""
    if pd.isna(temp_str):
        return None
    temp_str = str(temp_str).strip()
    match = re.match(r'(-?\d+\.?\d*)\s*C', temp_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def parse_concentration(conc_str):
    """
    Extract concentration from string like '.5 M/L', '0.75 M/L', '1.0 M/L'.
    Returns concentration in mol/L (will need to convert to molal later if needed).
    """
    if pd.isna(conc_str):
        return None
    conc_str = str(conc_str).strip()

    # Match patterns like ".5", "0.75", "1.0", "1.5"
    match = re.match(r'\.?(\d+\.?\d*)', conc_str)
    if match:
        return float(match.group(1))
    return None

def parse_conductivity(cond_str):
    """
    Extract conductivity value. Values appear to be in mS/cm based on typical ranges.
    Convert to S/m by dividing by 10.
    """
    if pd.isna(cond_str):
        return None
    try:
        val = float(cond_str)
        # Convert mS/cm to S/m: 1 mS/cm = 0.1 S/m
        return val / 10.0
    except (ValueError, TypeError):
        return None

def main():
    # Read the CSV
    df = pd.read_csv('../JesusData2025.csv', header=None)

    # Storage for parsed data
    records = []

    print("Parsing JesusData2025.csv...")
    print("=" * 70)

    # Manual parsing based on visual inspection of the data structure
    # This is a complex spreadsheet with multiple sections

    # Section 1: 1:1 NaCl:MgSO4 (rows 1-4, cols 2-5)
    print("\nParsing: 1:1 NaCl:MgSO4")
    concentrations = [0.5, 0.75, 1.0, 1.5]  # M/L
    for i, conc in enumerate(concentrations):
        col_idx = 2 + i  # Start at column 2
        for row_idx in [2, 3, 4]:  # Temperature rows
            temp = parse_temperature(df.iloc[row_idx, 1])
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if temp is not None and cond is not None:
                records.append({
                    'compound': 'NaCl:MgSO4_1:1',
                    'concentration_molar': conc,
                    'temperature_C': temp,
                    'conductivity_Sm': cond,
                    'notes': f'Original: {conc} M/L'
                })

    # Section 2: 2:1 NaCl:MgSO4 (1.5:0.75 M/L) (rows 8-10, col 1)
    print("Parsing: 2:1 NaCl:MgSO4 (1.5:0.75)")
    for row_idx in [8, 9, 10]:
        temp = parse_temperature(df.iloc[row_idx, 0])
        cond = parse_conductivity(df.iloc[row_idx, 1])
        if temp is not None and cond is not None:
            records.append({
                'compound': 'NaCl:MgSO4_2:1',
                'concentration_molar': 1.5,  # NaCl concentration
                'temperature_C': temp,
                'conductivity_Sm': cond,
                'notes': '1.5:0.75 M/L NaCl:MgSO4'
            })

    # Section 3: 1:2 NaCl:MgSO4 (0.75:1.5 M/L) (rows 8-10, col 4)
    print("Parsing: 1:2 NaCl:MgSO4 (0.75:1.5)")
    for row_idx in [8, 9, 10]:
        temp = parse_temperature(df.iloc[row_idx, 3])
        cond = parse_conductivity(df.iloc[row_idx, 4])
        if temp is not None and cond is not None:
            records.append({
                'compound': 'NaCl:MgSO4_1:2',
                'concentration_molar': 0.75,  # NaCl concentration
                'temperature_C': temp,
                'conductivity_Sm': cond,
                'notes': '0.75:1.5 M/L NaCl:MgSO4'
            })

    # Section 4: KCl only (rows 7-9, cols 7-10)
    print("Parsing: KCl")
    concentrations_kcl = [0.5, 0.75, 1.0, 1.5]
    for i, conc in enumerate(concentrations_kcl):
        col_idx = 7 + i
        for row_idx in [7, 8, 9]:  # Temperature rows
            temp = parse_temperature(df.iloc[row_idx, 6])
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if temp is not None and cond is not None:
                records.append({
                    'compound': 'KCl',
                    'concentration_molar': conc,
                    'temperature_C': temp,
                    'conductivity_Sm': cond,
                    'notes': f'{conc} M/L'
                })

    # Section 5: Na2SO4 only (rows 11-13, cols 7-9)
    print("Parsing: Na2SO4")
    concentrations_na2so4 = [0.5, 0.75]  # Only 2 concentrations visible
    for i, conc in enumerate(concentrations_na2so4):
        col_idx = 7 + i
        for row_idx in [11, 12, 13]:  # Temperature rows
            temp = parse_temperature(df.iloc[row_idx, 6])
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if temp is not None and cond is not None:
                records.append({
                    'compound': 'Na2SO4',
                    'concentration_molar': conc,
                    'temperature_C': temp,
                    'conductivity_Sm': cond,
                    'notes': f'{conc} M/L'
                })

    # Section 6: 1:1 Na2SO4:KCl (rows 14-16, cols 1-3)
    print("Parsing: 1:1 Na2SO4:KCl")
    concentrations_mix = [0.5, 0.75, 1.0]
    for i, conc in enumerate(concentrations_mix):
        col_idx = 1 + i
        for row_idx in [14, 15, 16]:
            temp = parse_temperature(df.iloc[row_idx, 0])
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if temp is not None and cond is not None:
                records.append({
                    'compound': 'Na2SO4:KCl_1:1',
                    'concentration_molar': conc,
                    'temperature_C': temp,
                    'conductivity_Sm': cond,
                    'notes': f'1:1 mixture, {conc} M/L each'
                })

    # Section 7: 2:1 Na2SO4:KCl (1.5:0.75) (rows 20-22, col 1)
    print("Parsing: 2:1 Na2SO4:KCl")
    for row_idx in [20, 21, 22]:
        temp = parse_temperature(df.iloc[row_idx, 0])
        cond = parse_conductivity(df.iloc[row_idx, 1])
        if temp is not None and cond is not None:
            records.append({
                'compound': 'Na2SO4:KCl_2:1',
                'concentration_molar': 1.5,
                'temperature_C': temp,
                'conductivity_Sm': cond,
                'notes': '1.5:0.75 M/L Na2SO4:KCl'
            })

    # Section 8: 1:2 Na2SO4:KCl (0.75:1.5) (rows 20-22, col 3)
    print("Parsing: 1:2 Na2SO4:KCl")
    for row_idx in [20, 21, 22]:
        temp = parse_temperature(df.iloc[row_idx, 2])
        cond = parse_conductivity(df.iloc[row_idx, 3])
        if temp is not None and cond is not None:
            records.append({
                'compound': 'Na2SO4:KCl_1:2',
                'concentration_molar': 0.75,
                'temperature_C': temp,
                'conductivity_Sm': cond,
                'notes': '0.75:1.5 M/L Na2SO4:KCl'
            })

    # Section 9: NaCl blanks (rows 147-152, cols 3-6)
    # These appear to be pure NaCl measurements (no glycine)
    print("Parsing: NaCl (blanks)")
    concentrations_nacl = [0.5, 0.75, 1.0, 1.5]  # M/L
    # Multiple replicates per concentration (6 rows of data)
    for i, conc in enumerate(concentrations_nacl):
        col_idx = 3 + i
        replicate_num = 1
        for row_idx in [147, 148, 149, 150, 151, 152]:
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if cond is not None:
                # Temperature not explicitly stated, appears to be 20°C based on context
                records.append({
                    'compound': 'NaCl',
                    'concentration_molar': conc,
                    'temperature_C': 20.0,
                    'conductivity_Sm': cond,
                    'notes': f'{conc} M/L, replicate {replicate_num}'
                })
                replicate_num += 1

    # Section 10: NaCl + Glycine (rows 153-158, cols 3-6)
    # NaCl with 100 mM/L glycine added
    print("Parsing: NaCl + Glycine")
    for i, conc in enumerate(concentrations_nacl):
        col_idx = 3 + i
        replicate_num = 1
        for row_idx in [153, 154, 155, 156, 157, 158]:
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if cond is not None:
                records.append({
                    'compound': 'NaCl+Glycine',
                    'concentration_molar': conc,
                    'temperature_C': 20.0,
                    'conductivity_Sm': cond,
                    'notes': f'{conc} M/L NaCl + 100mM/L glycine, replicate {replicate_num}'
                })
                replicate_num += 1

    # Section 11: Pure organic compounds (amino acids) - 100 mM/L each
    # Rows 161-169, cols 3-6 for Alanine, Aspartic Acid, Glutamic Acid, Glycine
    print("Parsing: Pure organic compounds (amino acids)")
    organics = ['Alanine', 'Aspartic Acid', 'Glutamic Acid', 'Glycine']
    for i, organic in enumerate(organics):
        col_idx = 3 + i
        replicate_num = 1
        for row_idx in [161, 162, 163, 164, 165, 166, 167, 168, 169]:
            cond = parse_conductivity(df.iloc[row_idx, col_idx])
            if cond is not None:
                # Concentration is 100 mM/L = 0.1 M/L
                records.append({
                    'compound': organic,
                    'concentration_molar': 0.1,
                    'temperature_C': 20.0,
                    'conductivity_Sm': cond,
                    'notes': f'100 mM/L {organic}, replicate {replicate_num}'
                })
                replicate_num += 1

    # Convert to DataFrame
    data_df = pd.DataFrame(records)

    # Add temperature in Kelvin
    data_df['temperature_K'] = data_df['temperature_C'] + 273.15

    # Add source
    data_df['source'] = 'benchtop'

    # For now, assume molar ≈ molal (valid for dilute solutions)
    # TODO: Proper conversion requires density data
    data_df['concentration_molal'] = data_df['concentration_molar']

    # Assign replicate numbers (group by compound, temp, conc and number them)
    # Some sections have explicit replicates tracked in notes, others don't
    def assign_replicate(group):
        # Check if replicates are already numbered in notes
        if 'replicate' in group['notes'].iloc[0]:
            # Extract replicate number from notes
            group['replicate'] = group['notes'].str.extract(r'replicate (\d+)')[0].astype(float)
        else:
            # Sequential numbering within group
            group['replicate'] = range(1, len(group) + 1)
        return group

    data_df = data_df.groupby(['compound', 'temperature_C', 'concentration_molar'], group_keys=False).apply(assign_replicate)
    data_df['replicate'] = data_df['replicate'].fillna(1).astype(int)

    # Reorder columns to match Mahboub format
    output_df = data_df[[
        'compound', 'concentration_molal', 'temperature_C', 'temperature_K',
        'conductivity_Sm', 'replicate', 'source', 'notes'
    ]]

    # Sort by compound, temperature, concentration
    output_df = output_df.sort_values(['compound', 'temperature_C', 'concentration_molal'])
    output_df = output_df.reset_index(drop=True)

    # Save to CSV
    output_file = 'Cortes2026BenchtopData.csv'

    # Create header comment
    header_lines = [
        "# Cortes et al. (2026) - Benchtop Conductivity Measurements",
        "#",
        "# Instrument: TBD (add details)",
        "# Reference: See Cortes et al. (2026) manuscript for full methodology",
        "#",
        "# Data description:",
        "# - All measurements from benchtop conductivity probe",
        "# - Gamry impedance data analyzed separately from raw Gamry files",
        "# - Temperatures: 5°C, 10°C, 20°C",
        "# - Compounds: KCl, NaCl, Na2SO4, NaCl:MgSO4 mixtures, Na2SO4:KCl mixtures,",
        "#              NaCl+Glycine, pure amino acids (Alanine, Aspartic Acid, Glutamic Acid, Glycine)",
        "# - Multiple replicates per condition for NaCl and organic compounds",
        "#",
        "# Columns:",
        "#   compound: Chemical formula or mixture notation",
        "#   concentration_molal: Molality (mol/kg_H2O)",
        "#     NOTE: Original data in M/L, assuming molal ≈ molar for dilute solutions",
        "#   temperature_C: Temperature (Celsius)",
        "#   temperature_K: Temperature (Kelvin)",
        "#   conductivity_Sm: Conductivity (S/m)",
        "#     NOTE: Converted from mS/cm by dividing by 10",
        "#   replicate: Replicate number",
        "#   source: Data source ('benchtop')",
        "#   notes: Additional notes",
        "#"
    ]

    with open(output_file, 'w') as f:
        f.write('\n'.join(header_lines) + '\n')
        output_df.to_csv(f, index=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS! Saved to: {output_file}")
    print("=" * 70)
    print(f"\nTotal measurements: {len(output_df)}")
    print(f"Compounds: {output_df['compound'].nunique()}")
    print(f"Temperature range: {output_df['temperature_C'].min():.1f} - {output_df['temperature_C'].max():.1f} °C")
    print(f"Concentration range: {output_df['concentration_molal'].min():.2f} - {output_df['concentration_molal'].max():.2f} mol/kg")
    print()

    print("Compound summary:")
    for compound in output_df['compound'].unique():
        comp_data = output_df[output_df['compound'] == compound]
        n_meas = len(comp_data)
        n_temps = comp_data['temperature_C'].nunique()
        n_concs = comp_data['concentration_molal'].nunique()
        print(f"  {compound:25s}: {n_meas:3d} measurements ({n_temps} temps × {n_concs} concs)")

    print()
    print("First 10 rows of output:")
    print(output_df.head(10).to_string())

    return output_df

if __name__ == '__main__':
    df = main()
