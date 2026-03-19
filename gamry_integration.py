#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamry impedance data integration for study plotting.

This module provides functions to load and process Gamry impedance analysis
results from HiPOZ workflow for overlay on benchtop conductivity plots.

Functions:
- load_gamry_results(): Load impedance results with error handling
- extract_compound_overlay(): Create overlay data for specific compound
- validate_gamry_data(): Check data completeness and quality

Usage:
    import gamry_integration as gi

    # Load Gamry results
    gamry_df = gi.load_gamry_results('data/20250815Mahboub2026')

    # Extract overlay for NaCl
    nacl_overlay = gi.extract_compound_overlay(gamry_df, 'NaCl')

    # Use in plotting
    if nacl_overlay:
        plot_study_concentration(..., gamry_data=nacl_overlay)
"""

import os
import glob
import numpy as np
import pandas as pd


def load_gamry_results(data_dir, verbose=True):
    """
    Load Gamry impedance analysis results from HiPOZ workflow.

    Searches for most recent hipoz_*_results.csv file in the data directory
    and loads it with comprehensive error handling.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing hipoz results
        (e.g., 'data/20250815Mahboub2026')
    verbose : bool, optional
        Print status messages and warnings (default: True)

    Returns
    -------
    gamry_df : pandas.DataFrame or None
        DataFrame with measurement data including conductivity_Sm column,
        or None if data unavailable/invalid

    Notes
    -----
    Error conditions handled:
    - Directory doesn't exist
    - No results files found
    - Missing conductivity_Sm column
    - All conductivity values are NaN
    - Partial missing conductivity (warning only)

    The function filters to measurements only (excludes standards).

    Examples
    --------
    >>> gamry_df = load_gamry_results('data/20250815Mahboub2026')
    Loading Gamry impedance results from hipoz_20260318_results.csv...
      Found 12 Gamry measurements
        MgSO4: 12 measurements

    >>> gamry_df = load_gamry_results('data/NoStandards')
      ⚠️  WARNING: No conductivity data in Gamry results
          Standards must be defined and associated with measurements.
    """
    gamry_df = None

    # Check if data directory exists
    if not os.path.exists(data_dir):
        if verbose:
            print(f"Note: Gamry data directory not found: {data_dir}")
            print("      Gamry overlays will be skipped.")
            print()
        return None

    # Look for most recent hipoz results file
    results_files = glob.glob(os.path.join(data_dir, 'hipoz_*_results.csv'))

    if not results_files:
        if verbose:
            print(f"Note: No HiPOZ results found in {data_dir}")
            print("      Run impedance analysis first:")
            print(f"      python gamry_HiPOZ.py --dates {os.path.basename(data_dir)}")
            print()
        return None

    # Use most recent results file
    results_file = max(results_files, key=os.path.getmtime)

    if verbose:
        print(f"Loading Gamry impedance results from {os.path.basename(results_file)}...")

    gamry_df = pd.read_csv(results_file)

    # Filter to measurements only (exclude standards)
    if 'type' in gamry_df.columns:
        gamry_df = gamry_df[gamry_df['type'] == 'measurement'].copy()

    # Check if conductivity data is available (standards defined and associated)
    if 'conductivity_Sm' not in gamry_df.columns:
        if verbose:
            print("  ⚠️  WARNING: No conductivity data in Gamry results")
            print("      Standards must be defined and associated with measurements.")
            config_file = f"zAnalysis{os.path.basename(data_dir)}.csv"
            print(f"      Check: {os.path.join(data_dir, config_file)}")
            print(f"      Then re-run: python gamry_HiPOZ.py --dates {os.path.basename(data_dir)}")
            print("      Gamry overlays will be skipped for this run.")
            print()
        return None

    elif gamry_df['conductivity_Sm'].isna().all():
        if verbose:
            print("  ⚠️  WARNING: All Gamry conductivity values are NaN")
            print("      This means standards are not properly assigned to measurements.")
            print("      In the HiPOZ GUI:")
            print("        1. Mark rows as 'Standard' (known conductivity)")
            print("        2. Select measurement rows")
            print("        3. Click 'Associate Measurements' to apply calibration")
            config_file = f"zAnalysis{os.path.basename(data_dir)}.csv"
            print(f"      Or edit: {os.path.join(data_dir, config_file)}")
            print(f"      Then re-run: python gamry_HiPOZ.py --dates {os.path.basename(data_dir)}")
            print("      Gamry overlays will be skipped for this run.")
            print()
        return None

    else:
        # Check for partially missing conductivity data
        n_total = len(gamry_df)
        n_valid = gamry_df['conductivity_Sm'].notna().sum()

        if verbose:
            print(f"  Found {len(gamry_df)} Gamry measurements")

            if n_valid < n_total:
                print(f"  ⚠️  WARNING: {n_total - n_valid}/{n_total} measurements missing conductivity")
                print("      Some measurements not associated with standards.")
                print("      Only measurements with conductivity will appear in plots.")
                print()

            if 'comp' in gamry_df.columns:
                for comp in gamry_df['comp'].dropna().unique():
                    n_comp = len(gamry_df[gamry_df['comp'] == comp])
                    n_valid_comp = gamry_df[gamry_df['comp'] == comp]['conductivity_Sm'].notna().sum()
                    if n_valid_comp < n_comp:
                        print(f"    {comp}: {n_valid_comp}/{n_comp} measurements with conductivity")
                    else:
                        print(f"    {comp}: {n_comp} measurements")
            print()

    return gamry_df


def extract_compound_overlay(gamry_df, compound, group_by='w_molal',
                              include_temp=True, verbose=False):
    """
    Extract Gamry overlay data for a specific compound.

    Filters DataFrame to specified compound, removes NaN conductivity values,
    groups by concentration, and calculates average conductivity with
    propagated uncertainties.

    Parameters
    ----------
    gamry_df : pandas.DataFrame or None
        Gamry results DataFrame from load_gamry_results()
    compound : str
        Compound name to filter (e.g., 'NaCl', 'MgSO4')
    group_by : str, optional
        Column to group by (default: 'w_molal' for molality)
        Use 'w_ppt' for g/kg or 'P_MPa' for pressure
    include_temp : bool, optional
        Include temperature in label (default: True)
    verbose : bool, optional
        Print debug information (default: False)

    Returns
    -------
    overlay_data : tuple or None
        If data available: (concentrations, conductivities, uncertainties, label)
        If no data: None

    Notes
    -----
    - Automatically filters out NaN conductivity values
    - Groups replicates by concentration and averages
    - Calculates RMS uncertainty: sqrt(sum(unc^2))/N for N replicates
    - Returns None if no valid data for compound

    Examples
    --------
    >>> gamry_df = load_gamry_results('data/20250815Mahboub2026')
    >>> nacl_overlay = extract_compound_overlay(gamry_df, 'NaCl')
    >>> if nacl_overlay:
    ...     conc, sigma, unc, label = nacl_overlay
    ...     print(f"{len(conc)} concentration points")
    2 concentration points

    >>> # Use in plotting
    >>> plot_study_concentration(..., gamry_data=nacl_overlay)
    """
    # Return None if input is None
    if gamry_df is None:
        return None

    # Check required columns exist
    if 'comp' not in gamry_df.columns:
        if verbose:
            print(f"Warning: 'comp' column not found in Gamry data")
        return None

    if group_by not in gamry_df.columns:
        if verbose:
            print(f"Warning: '{group_by}' column not found in Gamry data")
        return None

    # Filter to specified compound
    comp_data = gamry_df[gamry_df['comp'] == compound].copy()

    if len(comp_data) == 0:
        if verbose:
            print(f"No Gamry data found for {compound}")
        return None

    # Filter to rows with valid conductivity data (standards assigned)
    comp_data = comp_data[comp_data['conductivity_Sm'].notna()].copy()

    if len(comp_data) == 0:
        if verbose:
            print(f"No valid conductivity data for {compound} (all NaN)")
        return None

    # Group by concentration and average
    agg_dict = {
        'conductivity_Sm': 'mean',
        'conductivity_Sm_unc': lambda x: np.sqrt(np.sum(x**2))/len(x) if len(x) > 1 else x.iloc[0]
    }

    if include_temp and 'T_K' in comp_data.columns:
        agg_dict['T_K'] = 'mean'

    grouped = comp_data.groupby(group_by).agg(agg_dict).reset_index()

    if len(grouped) == 0:
        return None

    # Prepare overlay tuple
    concentrations = grouped[group_by].values
    conductivities = grouped['conductivity_Sm'].values
    uncertainties = grouped['conductivity_Sm_unc'].values

    # Create label
    if include_temp and 'T_K' in grouped.columns:
        temp_K = grouped['T_K'].iloc[0]
        label = f'Gamry ({temp_K:.2f} K)'
    else:
        label = 'Gamry'

    if verbose:
        print(f"Extracted {len(grouped)} {compound} overlay points")

    return (concentrations, conductivities, uncertainties, label)


def validate_gamry_data(gamry_df, required_columns=None, verbose=True):
    """
    Validate Gamry DataFrame for common issues.

    Parameters
    ----------
    gamry_df : pandas.DataFrame or None
        Gamry results DataFrame
    required_columns : list of str, optional
        Columns that must be present (default: standard set)
    verbose : bool, optional
        Print validation results (default: True)

    Returns
    -------
    is_valid : bool
        True if all validation checks pass

    Examples
    --------
    >>> gamry_df = load_gamry_results('data/20250815Mahboub2026')
    >>> is_valid = validate_gamry_data(gamry_df)
    ✅ Gamry data validation passed
        12 measurements
        Compounds: MgSO4
        Valid conductivity: 12/12
    """
    if required_columns is None:
        required_columns = ['comp', 'w_molal', 'conductivity_Sm',
                           'conductivity_Sm_unc', 'T_K']

    if gamry_df is None:
        if verbose:
            print("❌ Gamry data is None (not loaded)")
        return False

    issues = []

    # Check required columns
    for col in required_columns:
        if col not in gamry_df.columns:
            issues.append(f"Missing required column: {col}")

    # Check for data
    if len(gamry_df) == 0:
        issues.append("DataFrame is empty")

    # Check conductivity completeness
    if 'conductivity_Sm' in gamry_df.columns:
        n_total = len(gamry_df)
        n_valid = gamry_df['conductivity_Sm'].notna().sum()
        if n_valid == 0:
            issues.append("All conductivity values are NaN")

    if verbose:
        if issues:
            print("❌ Gamry data validation failed:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("✅ Gamry data validation passed")
            n_total = len(gamry_df)
            n_valid = gamry_df['conductivity_Sm'].notna().sum()
            print(f"    {n_total} measurements")
            if 'comp' in gamry_df.columns:
                compounds = gamry_df['comp'].dropna().unique()
                print(f"    Compounds: {', '.join(compounds)}")
            print(f"    Valid conductivity: {n_valid}/{n_total}")

    return len(issues) == 0


# Example usage
if __name__ == '__main__':
    print("Gamry Integration Module")
    print("=" * 70)
    print()
    print("This module provides functions to integrate Gamry impedance data")
    print("with benchtop conductivity measurements for study plotting.")
    print()
    print("Example usage:")
    print()
    print("    import gamry_integration as gi")
    print()
    print("    # Load Gamry results")
    print("    gamry_df = gi.load_gamry_results('data/20250815Mahboub2026')")
    print()
    print("    # Validate data")
    print("    if gi.validate_gamry_data(gamry_df):")
    print("        # Extract overlay for NaCl")
    print("        nacl_overlay = gi.extract_compound_overlay(gamry_df, 'NaCl')")
    print()
    print("        # Use in plotting")
    print("        if nacl_overlay:")
    print("            plot_study_concentration(..., gamry_data=nacl_overlay)")
    print()
    print("See function docstrings for detailed documentation.")
