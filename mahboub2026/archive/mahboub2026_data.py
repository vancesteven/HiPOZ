#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahboub et al. (2026, in press) Data Loading Utilities

This module provides functions specific to loading and organizing the
Mahboub2026BenchtopData.csv file containing benchtop conductivity probe
measurements. It uses the general plotting functions from plotting.py.

Citation:
    Mahboub, R., et al. (2026). [In press]

For future datasets, create similar data loading modules that output
the same structure for compatibility with plotting.py
"""

import os
import numpy as np
import pandas as pd
import logging
from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012

log = logging.getLogger('HiPOZ')


def compute_mccleskey_model(conc_molal, temps_K, compound):
    """
    Compute McCleskey et al. (2012) conductivity model at specified points.

    Parameters
    ----------
    conc_molal : ndarray
        Concentrations in mol/kg (molality)
    temps_K : ndarray
        Temperatures in Kelvin
    compound : str
        Compound name ('NaCl', 'MgSO4', 'NH4Cl', 'Na2CO3')

    Returns
    -------
    sigma_model : list of ndarray
        Conductivity at each temperature (S/m)
        Length M (number of temps), each array length N (number of concs)
    """
    # Ion specifications for each compound
    ion_specs = {
        'NaCl': {'Na_p1': 1.0, 'Cl_m1': 1.0},
        'MgSO4': {'Mg_p2': 1.0, 'SO4_m2': 1.0},
        'NH4Cl': {'NH4_p1': 1.0, 'Cl_m1': 1.0},
        'Na2CO3': {'Na_p1': 2.0, 'CO3_m2': 1.0}
    }

    if compound not in ion_specs:
        log.warning(f"No McCleskey model for {compound}")
        return []

    ionspec = ion_specs[compound]
    concs = np.asarray(conc_molal, dtype=float)
    temps_C = np.asarray(temps_K, dtype=float) - 273.15

    sigma_by_temp = []
    for T_C in temps_C:
        # Build ions dict for this temperature
        ions = {ion: {"mols": concs * mult} for ion, mult in ionspec.items()}

        # Call McCleskey model (pass scalar T_C)
        result = elecCondMcCleskey2012(float(T_C), ions)

        # Extract conductivity (converted to S/m in the function)
        # sigma_Sm has shape (n_temps, n_concs) where n_temps=1 for scalar T_C
        sigma_Sm = result['sigma_Sm']
        if len(sigma_Sm.shape) > 1:
            sigma_Sm = sigma_Sm[0, :]  # Extract the concentration array
        sigma_by_temp.append(sigma_Sm)

    return sigma_by_temp


def load_mahboub_benchtop_data(file_path='Mahboub2026BenchtopData.csv'):
    """
    Load benchtop conductivity data from Mahboub et al. 2026 study.

    Parameters
    ----------
    file_path : str
        Path to Mahboub2026BenchtopData.csv file

    Returns
    -------
    data : dict
        Dictionary with compound names as keys, each containing:
        - 'concentration_molal': concentrations in mol/kg
        - 'temperatures_K': temperatures in K
        - 'temperatures_C': temperatures in °C
        - 'conductivities_Sm': conductivity values in S/m
        - 'replicates': replicate number
        - 'source': data source (benchtop, Gamry, McCleskey2011)
    """
    # Try multiple possible paths
    possible_paths = [
        file_path,
        os.path.join(os.path.dirname(__file__), file_path),
        os.path.join(os.getcwd(), file_path)
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Read CSV, skipping comment lines
                df = pd.read_csv(path, comment='#')
                log.info(f"Loaded benchtop data from: {path}")
                break
            except Exception as e:
                log.warning(f"Failed to load {path}: {e}")
                continue

    if df is None:
        raise FileNotFoundError(f"Could not find Mahboub benchtop data file: {file_path}")

    # Organize by compound
    data = {}
    for compound in df['compound'].unique():
        comp_data = df[df['compound'] == compound].copy()

        data[compound] = {
            'concentration_molal': comp_data['concentration_molal'].values,
            'temperatures_K': comp_data['temperature_K'].values,
            'temperatures_C': comp_data['temperature_C'].values,
            'conductivities_Sm': comp_data['conductivity_Sm'].values,
            'replicates': comp_data['replicate'].values,
            'source': comp_data['source'].values
        }

    log.info(f"Loaded data for compounds: {list(data.keys())}")
    return data


def organize_by_temperature(data, compound, source_filter='benchtop'):
    """
    Organize Mahboub data by temperature for σ vs concentration plots.

    Uses general structure compatible with plotting.plot_sigma_vs_concentration()

    Parameters
    ----------
    data : dict
        Output from load_mahboub_benchtop_data()
    compound : str
        Compound name ('NaCl', 'MgSO4', etc.)
    source_filter : str or list
        Filter by source ('benchtop', 'Gamry20250813', 'McCleskey2011', etc.)

    Returns
    -------
    conc_array : ndarray
        Unique concentrations (mol/kg)
    sigma_by_temp : list of ndarray
        Conductivity at each temperature (averaged over replicates)
    errors_by_temp : list of ndarray
        Standard deviations
    temp_labels : list of str
        Temperature labels
    """
    comp_data = data[compound]

    # Filter by source
    if isinstance(source_filter, str):
        source_filter = [source_filter]

    mask = np.isin(comp_data['source'], source_filter)
    concs = comp_data['concentration_molal'][mask]
    temps_K = comp_data['temperatures_K'][mask]
    temps_C = comp_data['temperatures_C'][mask]
    sigmas = comp_data['conductivities_Sm'][mask]

    # Get unique values
    unique_temps_K = np.unique(temps_K)
    unique_concs = np.unique(concs)

    sigma_by_temp = []
    errors_by_temp = []
    temp_labels = []

    for T_K in unique_temps_K:
        T_C = temps_C[temps_K == T_K][0]
        temp_labels.append(f'{T_K:.2f} K')

        sigma_at_T = []
        for conc in unique_concs:
            mask_TC = (concs == conc) & (temps_K == T_K)
            sigma_at_TC = sigmas[mask_TC]
            sigma_at_T.append(np.mean(sigma_at_TC) if len(sigma_at_TC) > 0 else np.nan)

        # Calculate errors (std + systematic)
        errors_at_T = []
        for conc in unique_concs:
            mask_TC = (concs == conc) & (temps_K == T_K)
            sigma_at_TC = sigmas[mask_TC]

            if len(sigma_at_TC) > 1:
                std = np.std(sigma_at_TC)
                mean = np.mean(sigma_at_TC)
                # Mahboub et al. systematic errors: 0.5% + 2.89% + 0.5%
                total_err = np.sqrt(std**2 + (0.005*mean)**2 + (0.0289*mean)**2 + (0.005*mean)**2)
                errors_at_T.append(total_err)
            else:
                errors_at_T.append(0.03 * sigma_at_T[-1] if not np.isnan(sigma_at_T[-1]) else 0.0)

        sigma_by_temp.append(np.array(sigma_at_T))
        errors_by_temp.append(np.array(errors_at_T))

    return unique_concs, sigma_by_temp, errors_by_temp, temp_labels


def organize_by_concentration(data, compound, source_filter='benchtop'):
    """
    Organize Mahboub data by concentration for σ vs temperature plots.

    Uses general structure compatible with plotting.plot_sigma_vs_temperature()

    Parameters
    ----------
    data : dict
        Output from load_mahboub_benchtop_data()
    compound : str
        Compound name
    source_filter : str or list
        Filter by source

    Returns
    -------
    temp_array : ndarray
        Unique temperatures (K)
    sigma_by_conc : list of ndarray
        Conductivity at each concentration (averaged over replicates)
    errors_by_conc : list of ndarray
        Standard deviations
    conc_labels : list of str
        Concentration labels
    """
    comp_data = data[compound]

    # Filter by source
    if isinstance(source_filter, str):
        source_filter = [source_filter]

    mask = np.isin(comp_data['source'], source_filter)
    concs = comp_data['concentration_molal'][mask]
    temps_K = comp_data['temperatures_K'][mask]
    sigmas = comp_data['conductivities_Sm'][mask]

    # Get unique values
    unique_temps_K = np.unique(temps_K)
    unique_concs = np.unique(concs)

    sigma_by_conc = []
    errors_by_conc = []
    conc_labels = []

    for conc in unique_concs:
        conc_labels.append(f'{conc:.4f} mol/kg')

        sigma_at_conc = []
        for T_K in unique_temps_K:
            mask_TC = (concs == conc) & (temps_K == T_K)
            sigma_at_TC = sigmas[mask_TC]
            sigma_at_conc.append(np.mean(sigma_at_TC) if len(sigma_at_TC) > 0 else np.nan)

        # Calculate errors
        errors_at_conc = []
        for T_K in unique_temps_K:
            mask_TC = (concs == conc) & (temps_K == T_K)
            sigma_at_TC = sigmas[mask_TC]

            if len(sigma_at_TC) > 1:
                std = np.std(sigma_at_TC)
                mean = np.mean(sigma_at_TC)
                total_err = np.sqrt(std**2 + (0.005*mean)**2 + (0.0289*mean)**2 + (0.005*mean)**2)
                errors_at_conc.append(total_err)
            else:
                errors_at_conc.append(0.03 * sigma_at_conc[-1] if not np.isnan(sigma_at_conc[-1]) else 0.0)

        sigma_by_conc.append(np.array(sigma_at_conc))
        errors_by_conc.append(np.array(errors_at_conc))

    return unique_temps_K, sigma_by_conc, errors_by_conc, conc_labels


def load_gamry_from_hipoz(curated_file, compound=None):
    """
    Load Gamry impedance data from HiPOZ curated output.

    Parameters
    ----------
    curated_file : str
        Path to hipoz_*_curated.csv file
    compound : str, optional
        Filter by compound (e.g., 'NaCl')

    Returns
    -------
    data : dict
        Dictionary with:
        - 'concentration_molal': concentrations (mol/kg)
        - 'concentration_ppt': concentrations (g/kg solution)
        - 'temperatures_K': temperatures
        - 'conductivities_Sm': conductivities
        - 'errors': uncertainties
    """
    df = pd.read_csv(curated_file)

    if compound:
        df = df[df['Comp'] == compound]

    # Handle column name variations
    # Concentration: check for 'w (ppt)' or 'w(ppt)'
    conc_ppt_col = 'w (ppt)' if 'w (ppt)' in df.columns else 'w(ppt)'

    # Convert ppt to molal if compound is known
    # For NaCl: M = 58.44 g/mol, for MgSO4: M = 120.366 g/mol
    molar_masses = {
        'NaCl': 58.44,
        'MgSO4': 120.366,
        'NH4Cl': 53.491,
        'Na2CO3': 105.9888
    }

    conc_ppt = df[conc_ppt_col].values
    if compound and compound in molar_masses:
        M = molar_masses[compound]
        # molal = (ppt/M) / (1 + ppt/1000)
        conc_molal = (conc_ppt / M) / (1 + conc_ppt / 1000)
    else:
        conc_molal = None

    # Conductivity: check for 'S (S/m)' or 'sigma (S/m)'
    sigma_col = 'S (S/m)' if 'S (S/m)' in df.columns else 'sigma (S/m)'

    # Errors: check for 'S± (S/m)' or 'sigma_unc'
    if 'S± (S/m)' in df.columns:
        errors = df['S± (S/m)'].values
    elif 'sigma_unc' in df.columns:
        errors = df['sigma_unc'].values
    else:
        errors = None

    return {
        'concentration_molal': conc_molal,
        'concentration_ppt': conc_ppt,
        'temperatures_K': df['T (K)'].values,
        'conductivities_Sm': df[sigma_col].values,
        'errors': errors
    }
