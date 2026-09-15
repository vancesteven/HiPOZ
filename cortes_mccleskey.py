#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
McCleskey model integration for Cortes data.

Functions to compute McCleskey2012 conductivity model for low-pressure
Cortes data and calculate deviation metrics.
"""

import numpy as np
import pandas as pd
from study_plots import compute_mccleskey_model, ION_SPECS


# McCleskey (2012) applicability limits (mol/kg_H2O)
MCCLESKEY_LIMITS = {
    'NaCl': 0.9999,      # Up to 1.0 mol/kg
    'KCl': 0.9999,       # Up to 1.0 mol/kg (assumed similar to NaCl)
    'MgSO4': 0.01245,    # Up to ~0.012 mol/kg (low solubility)
    'Na2SO4': 0.3041,    # Up to ~0.3 mol/kg (assumed similar to Na2CO3)
    'NH4Cl': 1.034,      # Up to ~1.0 mol/kg
    'Na2CO3': 0.3041     # Up to ~0.3 mol/kg
}


def filter_low_pressure(data, p_threshold=5.0):
    """
    Filter data to low-pressure measurements suitable for McCleskey comparison.

    McCleskey model is only valid near atmospheric pressure (1 bar = 0.1 MPa).

    Parameters
    ----------
    data : pandas.DataFrame
        Measurement data with P_MPa column
    p_threshold : float
        Maximum pressure in MPa (default: 5.0 MPa)

    Returns
    -------
    pandas.DataFrame
        Filtered data with P_MPa <= p_threshold
    """
    if 'P_MPa' not in data.columns:
        print("Warning: No P_MPa column found. Cannot filter by pressure.")
        return data

    low_p_data = data[data['P_MPa'] <= p_threshold].copy()

    print(f"Filtered to low pressure (P ≤ {p_threshold} MPa):")
    print(f"  Before: {len(data)} measurements")
    print(f"  After: {len(low_p_data)} measurements")

    return low_p_data


def has_mccleskey_model(compound):
    """
    Check if McCleskey model is available for this compound.

    Parameters
    ----------
    compound : str
        Compound name (e.g., 'NaCl', 'KCl')

    Returns
    -------
    bool
        True if model is available
    """
    # Only single salts, not mixtures or organics
    if ',' in str(compound) or ':' in str(compound) or '+' in str(compound):
        return False

    return compound in ION_SPECS


def compute_mccleskey_for_data(data, compound):
    """
    Compute McCleskey model predictions for dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Measurement data with w_molal and T_K columns
    compound : str
        Compound name

    Returns
    -------
    model_sigma : ndarray
        Model predictions (S/m) for each row
    """
    if not has_mccleskey_model(compound):
        return None

    if 'w_molal' not in data.columns or 'T_K' not in data.columns:
        print(f"Warning: Missing w_molal or T_K columns for {compound}")
        return None

    # Get unique (concentration, temperature) pairs
    concs = data['w_molal'].values
    temps = data['T_K'].values

    model_sigma = np.zeros(len(data))

    for i, (conc, temp) in enumerate(zip(concs, temps)):
        try:
            # Compute model for single point
            result = compute_mccleskey_model([conc], [temp], compound=compound)
            model_sigma[i] = result[0][0]
        except Exception as e:
            print(f"Warning: McCleskey model failed for {compound} at {conc} mol/kg, {temp} K: {e}")
            model_sigma[i] = np.nan

    return model_sigma


def compute_delta(data_sigma, model_sigma):
    """
    Compute percent deviation from model.

    Delta = 100 * (data - model) / model

    Parameters
    ----------
    data_sigma : array-like
        Experimental conductivity (S/m)
    model_sigma : array-like
        Model predictions (S/m)

    Returns
    -------
    delta : ndarray
        Percent deviation
    """
    data_sigma = np.asarray(data_sigma, dtype=float)
    model_sigma = np.asarray(model_sigma, dtype=float)

    delta = np.full_like(data_sigma, np.nan, dtype=float)
    mask = np.isfinite(data_sigma) & np.isfinite(model_sigma) & (model_sigma != 0)
    delta[mask] = 100.0 * (data_sigma[mask] - model_sigma[mask]) / model_sigma[mask]

    return delta


def add_mccleskey_comparison(data, p_threshold=5.0):
    """
    Add McCleskey model and deviation columns to dataframe.

    Filters to low pressure and adds columns:
    - model_Sm: McCleskey prediction
    - delta_pct: Percent deviation

    Parameters
    ----------
    data : pandas.DataFrame
        Measurement data
    p_threshold : float
        Maximum pressure for McCleskey comparison (MPa)

    Returns
    -------
    pandas.DataFrame
        Data with added columns (only low-P rows kept)
    """
    # Filter to low pressure
    low_p = filter_low_pressure(data, p_threshold)

    if len(low_p) == 0:
        print("Warning: No low-pressure data found for McCleskey comparison")
        return low_p

    # Add model and delta columns
    low_p['model_Sm'] = np.nan
    low_p['delta_pct'] = np.nan

    # Compute for each compound
    for compound in low_p['comp'].dropna().unique():
        if not has_mccleskey_model(compound):
            print(f"Skipping {compound} - no McCleskey model available")
            continue

        mask = low_p['comp'] == compound
        subset = low_p[mask]

        model_sigma = compute_mccleskey_for_data(subset, compound)

        if model_sigma is not None:
            low_p.loc[mask, 'model_Sm'] = model_sigma

            # Compute delta
            delta = compute_delta(subset['conductivity_Sm'].values, model_sigma)
            low_p.loc[mask, 'delta_pct'] = delta

            n_valid = np.sum(np.isfinite(delta))
            print(f"  {compound}: {n_valid} points with McCleskey comparison")

    return low_p


def get_mccleskey_limit(compound):
    """
    Get McCleskey applicability limit for compound.

    Parameters
    ----------
    compound : str
        Compound name

    Returns
    -------
    float or None
        Concentration limit (mol/kg) or None if not available
    """
    return MCCLESKEY_LIMITS.get(compound, None)


if __name__ == '__main__':
    # Test script
    print("McCleskey Model Integration - Test")
    print("=" * 70)
    print()

    print("Available compounds:")
    for compound in ION_SPECS.keys():
        limit = get_mccleskey_limit(compound)
        print(f"  {compound:10s} - limit: {limit:.4f} mol/kg" if limit else f"  {compound:10s}")
    print()

    # Test computation
    import sys
    sys.path.insert(0, '.')
    from cortes_data_processing import load_cortes_data, average_replicates

    data = load_cortes_data(['20250813Cortes', '20250814Cortes'])
    if data is not None:
        print(f"Loaded {len(data)} measurements")

        # Test low-pressure filtering
        low_p = filter_low_pressure(data, p_threshold=5.0)
        print()

        # Test McCleskey addition
        with_model = add_mccleskey_comparison(data, p_threshold=5.0)
        print()
        print(f"Data with McCleskey comparison: {len(with_model)} rows")

        if 'delta_pct' in with_model.columns:
            n_valid = with_model['delta_pct'].notna().sum()
            print(f"Valid delta values: {n_valid}")

            if n_valid > 0:
                mean_delta = with_model['delta_pct'].mean()
                std_delta = with_model['delta_pct'].std()
                print(f"Mean delta: {mean_delta:.2f}%")
                print(f"Std delta: {std_delta:.2f}%")
