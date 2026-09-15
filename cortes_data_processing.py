#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing utilities for Cortes conductivity data.

Handles replicate averaging, grouping, and preparation for publication plots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from study_plots import compute_mccleskey_model, ION_SPECS


def load_cortes_data(data_dirs):
    """
    Load Gamry impedance data from zAnalysis CSV files.

    Parameters
    ----------
    data_dirs : list of str
        Data directory names (e.g., ['20250813Cortes', '20250814Cortes'])

    Returns
    -------
    pandas.DataFrame
        Combined measurement data with conductivity values
    """
    all_data = []

    for data_dir in data_dirs:
        dir_name = Path(data_dir).name
        date_only = dir_name[:8] if len(dir_name) >= 8 else dir_name

        csv_file = Path('data') / data_dir / f'zAnalysis{date_only}.csv'

        if not csv_file.exists():
            print(f"Warning: {csv_file} not found")
            continue

        df = pd.read_csv(csv_file)

        # Filter to measurements with conductivity
        df_meas = df[(df['type'] == 'measurement') &
                     (df['conductivity_Sm'].notna())].copy()

        if len(df_meas) > 0:
            all_data.append(df_meas)
            print(f"Loaded {len(df_meas)} measurements from {csv_file.name}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def average_replicates(data, p_tolerance=2.0, t_tolerance=0.5):
    """
    Average replicate measurements and calculate standard error of mean.

    Replicates are identified as measurements with:
    - Same compound (comp)
    - Same concentration (w_molal)
    - Similar pressure (within p_tolerance MPa)
    - Similar temperature (within t_tolerance K)

    Parameters
    ----------
    data : pandas.DataFrame
        Raw measurement data
    p_tolerance : float
        Pressure tolerance for grouping replicates (MPa)
    t_tolerance : float
        Temperature tolerance for grouping replicates (K)

    Returns
    -------
    pandas.DataFrame
        Averaged data with columns: comp, w_molal, w_ppt, P_MPa, T_K,
        conductivity_Sm (mean), conductivity_sem (SEM), n_replicates
    """
    # Round P and T to group replicates
    data = data.copy()
    data['P_group'] = (data['P_MPa'] / p_tolerance).round() * p_tolerance
    data['T_group'] = (data['T_K'] / t_tolerance).round() * t_tolerance

    # Group by composition and rounded P, T
    group_cols = ['comp', 'w_molal', 'w_ppt', 'P_group', 'T_group']

    # Build aggregation dict - only include S_unc_pct if it exists
    agg_dict = {
        'P_MPa': ['mean', 'std', 'count'],
        'T_K': ['mean', 'std'],
        'conductivity_Sm': ['mean', 'std', 'count']
    }

    if 'S_unc_pct' in data.columns:
        agg_dict['S_unc_pct'] = 'mean'

    grouped = data.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in grouped.columns.values]

    # Calculate SEM (standard error of mean)
    grouped['conductivity_sem'] = (grouped['conductivity_Sm_std'] /
                                   np.sqrt(grouped['conductivity_Sm_count']))

    # Use mean P and T as representative values
    grouped['P_MPa'] = grouped['P_MPa_mean']
    grouped['T_K'] = grouped['T_K_mean']

    # Keep only useful columns
    result_cols = [
        'comp', 'w_molal', 'w_ppt',
        'P_MPa', 'T_K',
        'conductivity_Sm_mean', 'conductivity_sem',
        'conductivity_Sm_count'
    ]

    if 'S_unc_pct' in grouped.columns:
        result_cols.append('S_unc_pct')

    result = grouped[result_cols].copy()

    # Rename for clarity
    result = result.rename(columns={
        'conductivity_Sm_mean': 'conductivity_Sm',
        'conductivity_Sm_count': 'n_replicates'
    })

    return result


def separate_by_composition(data):
    """
    Separate data into single salts and mixtures.

    Parameters
    ----------
    data : pandas.DataFrame
        Averaged data

    Returns
    -------
    dict
        Dictionary with keys 'single_salts' and 'mixtures', each containing
        a list of (composition_key, subset_data) tuples
    """
    single_salts = []
    mixtures = []

    # Group by unique (comp, w_molal) combinations
    for (comp, molal), group in data.groupby(['comp', 'w_molal'], dropna=False):
        if pd.isna(comp):
            continue

        # Check if mixture (comma in comp name)
        is_mixture = ',' in str(comp)

        # Create readable key
        if is_mixture:
            comps = str(comp).split(',')
            molals = str(molal).split(',') if ',' in str(molal) else [str(molal)]
            comp_key = '+'.join(comps) + '_' + '_'.join([f"{m}M" for m in molals])
        else:
            comp_key = f"{comp}_{float(molal):.2f}M".replace('.', 'p')

        if is_mixture:
            mixtures.append((comp_key, comp, molal, group))
        else:
            single_salts.append((comp_key, comp, molal, group))

    return {
        'single_salts': single_salts,
        'mixtures': mixtures
    }


def filter_by_pressure(data, p_max=5.0):
    """
    Filter data to low-pressure measurements (for McCleskey comparison).

    Parameters
    ----------
    data : pandas.DataFrame
        Data to filter
    p_max : float
        Maximum pressure (MPa)

    Returns
    -------
    pandas.DataFrame
        Filtered data with P <= p_max
    """
    return data[data['P_MPa'] <= p_max].copy()


if __name__ == '__main__':
    # Test with Cortes data
    print("Testing Cortes data processing...")
    print("=" * 70)

    data = load_cortes_data(['20250813Cortes', '20250814Cortes'])

    if data is not None:
        print(f"\nTotal raw measurements: {len(data)}")
        print(f"Unique compositions: {data['comp'].nunique()}")

        # Average replicates
        averaged = average_replicates(data)
        print(f"\nAfter averaging: {len(averaged)} data points")
        print(f"Average replicates per point: {averaged['n_replicates'].mean():.1f}")

        # Separate by composition
        separated = separate_by_composition(averaged)
        print(f"\nSingle salts: {len(separated['single_salts'])} compositions")
        print(f"Mixtures: {len(separated['mixtures'])} compositions")

        # Show example
        print("\nExample averaged data:")
        print(averaged[['comp', 'w_molal', 'P_MPa', 'T_K',
                       'conductivity_Sm', 'conductivity_sem', 'n_replicates']].head(10))
