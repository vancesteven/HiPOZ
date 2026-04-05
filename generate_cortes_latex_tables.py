#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables for Cortes data following supplement.tex format.

Creates tables similar to Mahboub supplement tables with:
- Conductivity data from impedance measurements
- Uncertainties propagated from impedance
- Standard calibration values
- Replicates shown
"""

import pandas as pd
import numpy as np
import cortes_data_processing as cdp
from pathlib import Path


def format_uncertainty(value, uncertainty, decimals=2):
    """
    Format value ± uncertainty for LaTeX.

    Parameters
    ----------
    value : float
        Central value
    uncertainty : float
        Uncertainty value
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string like "value $\\pm$ unc"
    """
    if pd.isna(uncertainty) or uncertainty == 0:
        return f"{value:.{decimals}f}"
    else:
        return f"{value:.{decimals}f} $\\pm$ {uncertainty:.{decimals}f}"


def generate_nacl_table(data, output_file='cortes_nacl_table.tex'):
    """
    Generate LaTeX table for NaCl data.

    Parameters
    ----------
    data : pandas.DataFrame
        Cortes data with all measurements
    output_file : str
        Output filename
    """
    # Filter to NaCl data only
    nacl_data = data[data['comp'] == 'NaCl'].copy()

    if len(nacl_data) == 0:
        print("No NaCl data found")
        return

    # Convert to numeric
    nacl_data['w_molal'] = pd.to_numeric(nacl_data['w_molal'], errors='coerce')
    nacl_data['T_K'] = pd.to_numeric(nacl_data['T_K'], errors='coerce')
    nacl_data['P_MPa'] = pd.to_numeric(nacl_data['P_MPa'], errors='coerce')

    # Sort by concentration, then temperature, then pressure
    nacl_data = nacl_data.sort_values(['w_molal', 'T_K', 'P_MPa'])

    # Find standard/calibration rows
    standards = nacl_data[nacl_data['type'] == 'standard']
    measurements = nacl_data[nacl_data['type'] == 'measurement']

    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write("$w$ (molal) NaCl & $T$ (K) & $P$ (MPa) & $\\sigma$ (\\si{S/m}) \\\\\n")
        f.write("\\hline\n")

        # Add standard rows first
        for idx, row in standards.iterrows():
            sigma_str = format_uncertainty(row['conductivity_Sm'],
                                          row.get('conductivity_Sm', 0) * row.get('S_unc_pct', 0) / 100,
                                          decimals=2)

            # Indicate this is a standard
            f.write(f"Standard & {row['T_K']:.0f} & {row['P_MPa']:.1f} & {sigma_str} \\\\\n")

        # Add measurement rows
        for idx, row in measurements.iterrows():
            if pd.isna(row['w_molal']):
                continue

            # Calculate sigma uncertainty from percentage
            if 'S_unc_pct' in row and pd.notna(row['S_unc_pct']):
                sigma_unc = row['conductivity_Sm'] * row['S_unc_pct'] / 100
            else:
                sigma_unc = np.nan

            sigma_str = format_uncertainty(row['conductivity_Sm'], sigma_unc, decimals=2)

            f.write(f"{row['w_molal']:.2f} & {row['T_K']:.0f} & {row['P_MPa']:.1f} & "
                   f"{sigma_str} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Conductivity of aqueous NaCl at high pressure computed from "
               "impedance spectroscopy. Gamry measurements from Cortes et al. (2026). "
               "Uncertainties propagated from impedance fitting.}\n")
        f.write("\\label{tab:NaClCortes}\n")
        f.write("\\end{table}\n")

    print(f"✓ Generated NaCl table: {output_file}")


def generate_mgso4_table(data, output_file='cortes_mgso4_table.tex'):
    """
    Generate LaTeX table for MgSO4 data.

    Parameters
    ----------
    data : pandas.DataFrame
        Cortes data with all measurements
    output_file : str
        Output filename
    """
    # Filter to MgSO4 data only
    mgso4_data = data[data['comp'] == 'MgSO4'].copy()

    if len(mgso4_data) == 0:
        print("No MgSO4 data found")
        return

    # Convert to numeric
    mgso4_data['w_molal'] = pd.to_numeric(mgso4_data['w_molal'], errors='coerce')
    mgso4_data['T_K'] = pd.to_numeric(mgso4_data['T_K'], errors='coerce')
    mgso4_data['P_MPa'] = pd.to_numeric(mgso4_data['P_MPa'], errors='coerce')

    # Sort by concentration, then temperature
    mgso4_data = mgso4_data.sort_values(['w_molal', 'T_K', 'P_MPa'])

    # Find standard/calibration rows
    standards = mgso4_data[mgso4_data['type'] == 'standard']
    measurements = mgso4_data[mgso4_data['type'] == 'measurement']

    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write("$w$ (molal) \\ce{MgSO4} & $T$ (K) & $P$ (MPa) & $\\sigma$ (\\si{S/m}) \\\\\n")
        f.write("\\hline\n")

        # Add standard rows first
        for idx, row in standards.iterrows():
            sigma_str = format_uncertainty(row['conductivity_Sm'],
                                          row.get('conductivity_Sm', 0) * row.get('S_unc_pct', 0) / 100,
                                          decimals=3)

            f.write(f"Standard & {row['T_K']:.0f} & {row['P_MPa']:.1f} & {sigma_str} \\\\\n")

        # Add measurement rows
        for idx, row in measurements.iterrows():
            if pd.isna(row['w_molal']):
                continue

            # Calculate sigma uncertainty from percentage
            if 'S_unc_pct' in row and pd.notna(row['S_unc_pct']):
                sigma_unc = row['conductivity_Sm'] * row['S_unc_pct'] / 100
            else:
                sigma_unc = np.nan

            sigma_str = format_uncertainty(row['conductivity_Sm'], sigma_unc, decimals=3)

            f.write(f"{row['w_molal']:.2f} & {row['T_K']:.0f} & {row['P_MPa']:.1f} & "
                   f"{sigma_str} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Conductivity of aqueous \\ce{MgSO4} at high pressure computed from "
               "impedance spectroscopy. Gamry measurements from Cortes et al. (2026). "
               "Uncertainties propagated from impedance fitting.}\n")
        f.write("\\label{tab:MgSO4Cortes}\n")
        f.write("\\end{table}\n")

    print(f"✓ Generated MgSO4 table: {output_file}")


def generate_mixture_table(data, output_file='cortes_mixture_table.tex'):
    """
    Generate LaTeX table for NaCl+MgSO4 mixture data.

    Parameters
    ----------
    data : pandas.DataFrame
        Cortes data with all measurements
    output_file : str
        Output filename
    """
    # Filter to mixture data only
    mixture_data = data[data['comp'].str.contains(',', na=False)].copy()

    if len(mixture_data) == 0:
        print("No mixture data found")
        return

    # Sort by composition
    mixture_data = mixture_data.sort_values(['comp', 'w_molal', 'T_K'])

    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\hline\n")
        f.write("$w$ (molal) NaCl,MgSO4 & $T$ (K) & $P$ (MPa) & $Z$ (\\si{\\ohm}) & $\\sigma$ (\\si{S/m}) \\\\\n")
        f.write("\\hline\n")

        # Add measurement rows
        for idx, row in mixture_data.iterrows():
            if pd.isna(row['w_molal']):
                continue

            z_str = format_uncertainty(row['Z_Ohm'], row.get('Z_unc_Ohm', np.nan), decimals=3)

            # Calculate sigma uncertainty from percentage
            if 'S_unc_pct' in row and pd.notna(row['S_unc_pct']):
                sigma_unc = row['conductivity_Sm'] * row['S_unc_pct'] / 100
            else:
                sigma_unc = np.nan

            sigma_str = format_uncertainty(row['conductivity_Sm'], sigma_unc, decimals=3)

            f.write(f"{row['w_molal']} & {row['T_K']:.0f} & {row['P_MPa']:.1f} & "
                   f"{z_str} & {sigma_str} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Conductivity of aqueous NaCl+\\ce{MgSO4} mixtures at atmospheric pressure "
               "computed from impedance spectroscopy. Gamry measurements from Cortes et al. (2026). "
               "Molality given as (NaCl, \\ce{MgSO4}). Uncertainties propagated from impedance fitting.}\n")
        f.write("\\label{tab:MixtureCortes}\n")
        f.write("\\end{table}\n")

    print(f"✓ Generated mixture table: {output_file}")


def generate_mccleskey_comparison_table(low_p_data, output_file='cortes_mccleskey_table.tex'):
    """
    Generate LaTeX table comparing data to McCleskey model.

    Parameters
    ----------
    low_p_data : pandas.DataFrame
        Low-pressure data with McCleskey model predictions
    output_file : str
        Output filename
    """
    if 'model_Sm' not in low_p_data.columns:
        print("No McCleskey model data found")
        return

    # Filter to only rows with model data
    comparison_data = low_p_data[low_p_data['model_Sm'].notna()].copy()

    if len(comparison_data) == 0:
        print("No McCleskey comparison data found")
        return

    # Convert to numeric
    comparison_data['w_molal'] = pd.to_numeric(comparison_data['w_molal'], errors='coerce')
    comparison_data['T_K'] = pd.to_numeric(comparison_data['T_K'], errors='coerce')

    # Sort by compound, concentration, temperature
    comparison_data = comparison_data.sort_values(['comp', 'w_molal', 'T_K'])

    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\hline\n")
        f.write("Compound & $w$ (molal) & $T$ (K) & $\\sigma_{exp}$ (\\si{S/m}) & "
               "$\\sigma_{McCleskey}$ (\\si{S/m}) & $\\Delta$ (\\%) \\\\\n")
        f.write("\\hline\n")

        # Add comparison rows
        for idx, row in comparison_data.iterrows():
            compound = row['comp']
            molal = row['w_molal']
            temp = row['T_K']
            sigma_exp = row['conductivity_Sm']
            sigma_model = row['model_Sm']
            delta = row['delta_pct']

            f.write(f"{compound} & {molal:.2f} & {temp:.1f} & {sigma_exp:.3f} & "
                   f"{sigma_model:.3f} & {delta:.2f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparison of experimental conductivity (Cortes et al. 2026) "
               "with McCleskey (2012) model predictions at low pressure (P $\\leq$ 5 MPa). "
               "Delta shows percent deviation: $\\Delta = 100 \\times (\\sigma_{exp} - \\sigma_{model}) / \\sigma_{model}$.}\n")
        f.write("\\label{tab:McCleskeyComparison}\n")
        f.write("\\end{table}\n")

    print(f"✓ Generated McCleskey comparison table: {output_file}")


def generate_benchtop_table(benchtop_file, output_file='cortes_benchtop_table.tex'):
    """
    Generate LaTeX table for benchtop conductivity data from JesusData2025.csv.

    The benchtop data has a complex sparse format with:
    - Multiple compound types (NaCl:MgSO4, KCl, Na2SO4, Glycine, etc.)
    - Multiple concentrations
    - Measurements at different temperatures (5C, 10C, 20C)
    - Values in mS/cm (needs conversion to S/m by dividing by 10)

    Parameters
    ----------
    benchtop_file : Path or str
        Path to JesusData2025.csv file
    output_file : str or Path
        Output LaTeX file path
    """
    print(f"Parsing benchtop data from: {benchtop_file}")

    # TODO: Implement proper parsing of JesusData2025.csv
    # The CSV has a complex sparse format that needs custom parsing
    # For now, creating a placeholder table

    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write("Compound & $w$ (M/L) & $T$ (\\si{\\degreeCelsius}) & $\\sigma$ (\\si{S/m}) \\\\\n")
        f.write("\\hline\n")

        # Placeholder - proper implementation needed
        f.write("% TODO: Parse JesusData2025.csv for benchtop measurements\n")
        f.write("% Format: Compound & Concentration & Temperature & Conductivity\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Benchtop conductivity measurements from Cortes et al. (2025). "
               "Measurements performed at atmospheric pressure. Values represent mean of replicates.}\n")
        f.write("\\label{tab:BenchtopCortes}\n")
        f.write("\\end{table}\n")

    print(f"⚠ Generated placeholder benchtop table: {output_file}")
    print(f"   Proper parsing of JesusData2025.csv needs implementation")


def main():
    """
    Generate all LaTeX tables for Cortes data.
    """
    print("=" * 70)
    print("Generating LaTeX Tables for Cortes Data")
    print("=" * 70)
    print()

    # Load Cortes data
    print("Loading data...")
    dates = ['20250813Cortes', '20250814Cortes', '20250815Cortes']
    data = cdp.load_cortes_data(dates)

    if data is None or len(data) == 0:
        print("ERROR: No data loaded")
        return

    print(f"Loaded {len(data)} measurements")
    print()

    # Create output directory
    output_dir = Path('latex_tables')
    output_dir.mkdir(exist_ok=True)

    # Generate tables
    print("Generating tables...")
    print("-" * 70)

    # 1. NaCl table
    generate_nacl_table(data, output_dir / 'cortes_nacl_table.tex')

    # 2. MgSO4 table
    generate_mgso4_table(data, output_dir / 'cortes_mgso4_table.tex')

    # 3. Mixture table
    generate_mixture_table(data, output_dir / 'cortes_mixture_table.tex')

    # 4. Benchtop data table (from Jesus Cortes original data)
    print()
    print("Generating benchtop data table...")
    try:
        benchtop_file = Path('JesusData2025.csv')
        if benchtop_file.exists():
            generate_benchtop_table(benchtop_file, output_dir / 'cortes_benchtop_table.tex')
        else:
            print(f"Warning: Benchtop data file not found: {benchtop_file}")
    except Exception as e:
        print(f"Warning: Could not generate benchtop table: {e}")

    # Note: McCleskey comparison tables removed per user request

    print()
    print("=" * 70)
    print(f"Complete! Tables saved to: {output_dir}/")
    print("=" * 70)
    print()
    print("Usage in LaTeX:")
    print("  \\input{latex_tables/cortes_nacl_table.tex}")
    print("  \\input{latex_tables/cortes_mgso4_table.tex}")
    print("  \\input{latex_tables/cortes_mixture_table.tex}")
    print("  \\input{latex_tables/cortes_mccleskey_table.tex}")
    print()


if __name__ == '__main__':
    main()
