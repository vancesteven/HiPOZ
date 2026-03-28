#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized plotting functions for conductivity studies.

This module provides reusable functions for generating publication-quality
conductivity plots from benchtop and Gamry impedance data. Designed to work
with any conductivity study (Mahboub 2026, Cortes et al., etc.).

Functions
---------
plot_study_concentration : Generate σ vs concentration plot with Delta subplot
plot_study_temperature : Generate σ vs temperature plot with Delta subplot
load_study_data : Load benchtop data from CSV
organize_for_conc_plot : Organize data by temperature
organize_for_temp_plot : Organize data by concentration
compute_model : Compute McCleskey model at data points
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012
from plotting import (
    plot_sigma_vs_concentration,
    plot_sigma_vs_temperature
)


# Ion specifications for McCleskey model
ION_SPECS = {
    'NaCl': {'Na_p1': 1.0, 'Cl_m1': 1.0},
    'MgSO4': {'Mg_p2': 1.0, 'SO4_m2': 1.0},
    'NH4Cl': {'NH4_p1': 1.0, 'Cl_m1': 1.0},
    'Na2CO3': {'Na_p1': 2.0, 'CO3_m2': 1.0}
}


def compute_mccleskey_model(conc_molal, temps_K, compound=None, ion_spec=None):
    """
    Compute McCleskey et al. (2012) conductivity model.

    Parameters
    ----------
    conc_molal : array-like
        Concentrations in mol/kg (molality)
    temps_K : array-like
        Temperatures in Kelvin
    compound : str, optional
        Compound name ('NaCl', 'MgSO4', 'NH4Cl', 'Na2CO3')
        Used to look up ion_spec if not provided
    ion_spec : dict, optional
        Custom ion specification, e.g. {'Na_p1': 1.0, 'Cl_m1': 1.0}
        If provided, overrides compound lookup

    Returns
    -------
    sigma_model : list of ndarray
        Conductivity at each temperature (S/m)
        Length M (number of temps), each array length N (number of concs)
    """
    if ion_spec is None:
        if compound is None:
            raise ValueError("Must provide either compound or ion_spec")
        if compound not in ION_SPECS:
            raise ValueError(f"No ion spec for {compound}. Provide custom ion_spec.")
        ion_spec = ION_SPECS[compound]

    concs = np.asarray(conc_molal, dtype=float)
    temps_C = np.asarray(temps_K, dtype=float) - 273.15

    sigma_by_temp = []
    for T_C in temps_C:
        # Build ions dict for this temperature
        ions = {ion: {"mols": concs * mult} for ion, mult in ion_spec.items()}

        # Call McCleskey model
        result = elecCondMcCleskey2012(float(T_C), ions)

        # Extract conductivity (converted to S/m in the function)
        sigma_Sm = result['sigma_Sm']
        if len(sigma_Sm.shape) > 1:
            sigma_Sm = sigma_Sm[0, :]  # Extract concentration array
        sigma_by_temp.append(sigma_Sm)

    return sigma_by_temp


def load_study_data(csv_file):
    """
    Load benchtop data from CSV file.

    Parameters
    ----------
    csv_file : str
        Path to CSV file with columns:
        compound, concentration_molal, temperature_C, temperature_K,
        conductivity_Sm, replicate, source, notes

    Returns
    -------
    data : dict
        Dictionary with compound names as keys, each containing arrays:
        - concentration_molal
        - temperatures_K
        - temperatures_C
        - conductivities_Sm
        - replicates
        - source
    """
    df = pd.read_csv(csv_file, comment='#')

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

    return data


def organize_for_conc_plot(data, compound, source_filter='benchtop',
                          filter_frozen=True):
    """
    Organize data by temperature for σ vs concentration plots.

    Parameters
    ----------
    data : dict
        Output from load_study_data()
    compound : str
        Compound name
    source_filter : str or list
        Filter by source ('benchtop', 'Gamry...', etc.)
    filter_frozen : bool
        If True, set zero/near-zero conductivity values to NaN (frozen samples)

    Returns
    -------
    conc_array : ndarray
        Unique concentrations
    sigma_by_temp : list of ndarray
        Conductivity at each temperature
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
        errors_at_T = []
        for conc in unique_concs:
            mask_TC = (concs == conc) & (temps_K == T_K)
            sigma_at_TC = sigmas[mask_TC]

            if len(sigma_at_TC) > 0:
                mean_val = np.mean(sigma_at_TC)

                # Filter frozen samples (zero or near-zero conductivity)
                if filter_frozen and abs(mean_val) < 1e-12:
                    sigma_at_T.append(np.nan)
                    errors_at_T.append(0.0)
                else:
                    sigma_at_T.append(mean_val)

                    if len(sigma_at_TC) > 1:
                        std = np.std(sigma_at_TC)
                        # Systematic errors: 0.5% + 2.89% + 0.5%
                        total_err = np.sqrt(std**2 + (0.005*mean_val)**2 +
                                           (0.0289*mean_val)**2 + (0.005*mean_val)**2)
                        errors_at_T.append(total_err)
                    else:
                        errors_at_T.append(0.03 * mean_val if mean_val > 1e-12 else 0.0)
            else:
                sigma_at_T.append(np.nan)
                errors_at_T.append(0.0)

        sigma_by_temp.append(np.array(sigma_at_T))
        errors_by_temp.append(np.array(errors_at_T))

    return unique_concs, sigma_by_temp, errors_by_temp, temp_labels


def organize_for_temp_plot(data, compound, source_filter='benchtop',
                          filter_frozen=True):
    """
    Organize data by concentration for σ vs temperature plots.

    Parameters
    ----------
    data : dict
        Output from load_study_data()
    compound : str
        Compound name
    source_filter : str or list
        Filter by source
    filter_frozen : bool
        If True, set zero/near-zero conductivity values to NaN (frozen samples)

    Returns
    -------
    temp_array : ndarray
        Unique temperatures (K)
    sigma_by_conc : list of ndarray
        Conductivity at each concentration
    errors_by_conc : list of ndarray
        Standard deviations
    conc_labels : list of str
        Concentration labels
    """
    comp_data = data[compound]

    if isinstance(source_filter, str):
        source_filter = [source_filter]

    mask = np.isin(comp_data['source'], source_filter)
    concs = comp_data['concentration_molal'][mask]
    temps_K = comp_data['temperatures_K'][mask]
    sigmas = comp_data['conductivities_Sm'][mask]

    unique_temps_K = np.unique(temps_K)
    unique_concs = np.unique(concs)

    sigma_by_conc = []
    errors_by_conc = []
    conc_labels = []

    for conc in unique_concs:
        conc_labels.append(f'{conc:.4f}')  # Just concentration value, no units for T plots

        sigma_at_conc = []
        errors_at_conc = []
        for T_K in unique_temps_K:
            mask_TC = (concs == conc) & (temps_K == T_K)
            sigma_at_TC = sigmas[mask_TC]

            if len(sigma_at_TC) > 0:
                mean_val = np.mean(sigma_at_TC)

                # Filter frozen samples (zero or near-zero conductivity)
                if filter_frozen and abs(mean_val) < 1e-12:
                    sigma_at_conc.append(np.nan)
                    errors_at_conc.append(0.0)
                else:
                    sigma_at_conc.append(mean_val)

                    if len(sigma_at_TC) > 1:
                        std = np.std(sigma_at_TC)
                        total_err = np.sqrt(std**2 + (0.005*mean_val)**2 +
                                           (0.0289*mean_val)**2 + (0.005*mean_val)**2)
                        errors_at_conc.append(total_err)
                    else:
                        errors_at_conc.append(0.03 * mean_val if mean_val > 1e-12 else 0.0)
            else:
                sigma_at_conc.append(np.nan)
                errors_at_conc.append(0.0)

        sigma_by_conc.append(np.array(sigma_at_conc))
        errors_by_conc.append(np.array(errors_at_conc))

    return unique_temps_K, sigma_by_conc, errors_by_conc, conc_labels


def plot_study_concentration(data, compound, output_file,
                            gamry_data=None, gamry_source='Gamry',
                            show_delta=True, show_legend=False, show_title=True,
                            compound_latex=None, mccleskey_limit=None, colormap='tab10',
                            fontsize_label=14, fontsize_title=16, fontsize_legend=10):
    """
    Generate σ vs concentration plot with optional Gamry overlay and Delta subplot.

    Parameters
    ----------
    data : dict
        Loaded study data from load_study_data()
    compound : str
        Compound name
    output_file : str
        Output PDF filename
    gamry_data : tuple, optional
        (concentrations, sigmas, errors, temperature_label) for Gamry overlay
        If None, will attempt to load from data using gamry_source filter
    gamry_source : str
        Source filter for Gamry data (default: 'Gamry')
    show_delta : bool
        Include deviation subplot
    show_legend : bool
        Include legend (default: False)
    show_title : bool
        Include title (default: True)
    compound_latex : str, optional
        LaTeX formatted compound name for title (e.g., r'MgSO$_4$')
    mccleskey_limit : float, optional
        McCleskey applicability limit (mol/kg) - draws vertical line
    fontsize_label, fontsize_title, fontsize_legend : int
        Font sizes

    Returns
    -------
    fig : matplotlib Figure
    """
    # Get benchtop data
    concs, sigmas, errors, temp_labels = organize_for_conc_plot(data, compound)

    # Extract unique temperatures
    comp_data = data[compound]
    benchtop_mask = comp_data['source'] == 'benchtop'
    unique_temps_K = np.unique(comp_data['temperatures_K'][benchtop_mask])

    # Compute McCleskey model if needed
    model_data = None
    if show_delta:
        try:
            model_data = compute_mccleskey_model(concs, unique_temps_K, compound=compound)
        except ValueError:
            # No model available for this compound
            model_data = None
            show_delta = False

    # Plot using general function
    if compound_latex is None:
        compound_latex = compound

    fig = plot_sigma_vs_concentration(
        conc_data=concs,
        sigma_data=sigmas,
        sigma_errors=errors,
        temp_labels=temp_labels,
        model_data=model_data if show_delta else None,
        xlabel=r'Concentration (mol/kg$_{\mathrm{H_2O}}$)',
        title=f'{compound_latex} Conductivity vs Concentration',
        show_delta=show_delta,
        show_title=show_title,
        limit=mccleskey_limit,
        out_file=output_file,
        colormap=colormap,
        fontsize_label=fontsize_label,
        fontsize_title=fontsize_title,
        fontsize_legend=fontsize_legend
    )

    # Overlay Gamry data if available
    if gamry_data is not None:
        g_conc, g_sigma, g_err, g_label = gamry_data
        ax = fig.axes[0]

        # Get colormap to match temperature color - use discrete indices like original
        cmap = mpl_cm.get_cmap(colormap)
        colors = cmap(np.arange(len(temp_labels)))  # Discrete colors for each temperature
        gamry_color = colors[5] if len(colors) > 5 else colors[-1]  # Default to ~20°C color (index 5)

        ax.errorbar(g_conc, g_sigma, yerr=g_err,
                   fmt='o', color='black', mfc=gamry_color, mec='black',
                   ms=10, capsize=5, linewidth=2, label=g_label)
        if show_legend:
            ax.legend(title='Temperature', fontsize=fontsize_legend)

        # Re-save figure with Gamry overlay
        fig.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig


def plot_study_temperature(data, compound, output_file,
                          gamry_data=None, show_delta=True,
                          show_legend=False, show_title=True,
                          compound_latex=None, colormap='tab10',
                          fontsize_label=14, fontsize_title=16, fontsize_legend=10):
    """
    Generate σ vs temperature plot with Delta subplot.

    Parameters
    ----------
    data : dict
        Loaded study data
    compound : str
        Compound name
    output_file : str
        Output PDF filename
    gamry_data : tuple, optional
        (concentrations, sigmas, errors, temperature_K) for Gamry overlay
    show_delta : bool
        Include deviation subplot
    compound_latex : str, optional
        LaTeX formatted compound name
    fontsize_label, fontsize_title, fontsize_legend : int
        Font sizes

    Returns
    -------
    fig : matplotlib Figure
    """
    # Get benchtop data
    temps, sigmas, errors, conc_labels = organize_for_temp_plot(data, compound)

    # Get unique concentrations
    comp_data = data[compound]
    benchtop_mask = comp_data['source'] == 'benchtop'
    unique_concs = np.unique(comp_data['concentration_molal'][benchtop_mask])

    # Compute McCleskey model if needed
    model_by_conc = None
    if show_delta:
        try:
            # Compute McCleskey model (transposed: one array per concentration)
            model_by_temp = compute_mccleskey_model(unique_concs, temps, compound=compound)
            # Transpose to get one array per concentration
            model_by_conc = []
            for i in range(len(unique_concs)):
                model_by_conc.append(np.array([model_by_temp[j][i] for j in range(len(temps))]))
        except ValueError:
            # No model available for this compound
            model_by_conc = None
            show_delta = False

    if compound_latex is None:
        compound_latex = compound

    fig = plot_sigma_vs_temperature(
        temp_data=temps,
        sigma_data=sigmas,
        sigma_errors=errors,
        conc_labels=conc_labels,
        model_data=model_by_conc,
        title=f'{compound_latex} Conductivity vs Temperature',
        show_delta=show_delta,
        show_title=show_title,
        out_file=output_file,
        colormap=colormap,
        fontsize_label=fontsize_label,
        fontsize_title=fontsize_title,
        fontsize_legend=fontsize_legend
    )

    return fig


def organize_for_pressure_plot(data, compound, source_filter='benchtop'):
    """
    Organize data by concentration for σ vs pressure plots.

    Parameters
    ----------
    data : dict
        Output from load_study_data()
    compound : str
        Compound name
    source_filter : str or list
        Filter by source

    Returns
    -------
    pressure_array : ndarray
        Unique pressures (MPa)
    sigma_by_conc : list of ndarray
        Conductivity at each concentration
    errors_by_conc : list of ndarray
        Standard deviations
    conc_labels : list of str
        Concentration labels
    """
    comp_data = data[compound]

    if isinstance(source_filter, str):
        source_filter = [source_filter]

    mask = np.isin(comp_data['source'], source_filter)
    concs = comp_data['concentration_molal'][mask]

    # Handle pressure data if available (otherwise skip)
    if 'pressure_MPa' not in comp_data:
        raise ValueError(f"No pressure data available for {compound}")

    pressures = comp_data['pressure_MPa'][mask]
    sigmas = comp_data['conductivities_Sm'][mask]

    unique_pressures = np.unique(pressures)
    unique_concs = np.unique(concs)

    sigma_by_conc = []
    errors_by_conc = []
    conc_labels = []

    for conc in unique_concs:
        conc_labels.append(f'{conc:.4f} mol/kg')

        sigma_at_conc = []
        errors_at_conc = []
        for P in unique_pressures:
            mask_PC = (concs == conc) & (pressures == P)
            sigma_at_PC = sigmas[mask_PC]

            if len(sigma_at_PC) > 0:
                mean_val = np.mean(sigma_at_PC)
                sigma_at_conc.append(mean_val)

                if len(sigma_at_PC) > 1:
                    std = np.std(sigma_at_PC)
                    total_err = np.sqrt(std**2 + (0.005*mean_val)**2 +
                                       (0.0289*mean_val)**2 + (0.005*mean_val)**2)
                    errors_at_conc.append(total_err)
                else:
                    errors_at_conc.append(0.03 * mean_val)
            else:
                sigma_at_conc.append(np.nan)
                errors_at_conc.append(0.0)

        sigma_by_conc.append(np.array(sigma_at_conc))
        errors_by_conc.append(np.array(errors_at_conc))

    return unique_pressures, sigma_by_conc, errors_by_conc, conc_labels


def plot_study_pressure(data, compound, output_file,
                       gamry_data=None, show_delta=False,
                       show_legend=False, show_title=True,
                       compound_latex=None, colormap='tab10',
                       fontsize_label=14, fontsize_title=16, fontsize_legend=10):
    """
    Generate σ vs pressure plot following the form of temperature plots.

    Parameters
    ----------
    data : dict
        Loaded study data
    compound : str
        Compound name
    output_file : str
        Output PDF filename
    gamry_data : tuple, optional
        (concentrations, sigmas, errors, pressure_MPa) for Gamry overlay
    show_delta : bool
        Include deviation subplot (model comparison)
    show_legend : bool
        Include legend (default: False)
    show_title : bool
        Include title (default: True)
    compound_latex : str, optional
        LaTeX formatted compound name
    colormap : str
        Matplotlib colormap name (default: 'tab10')
    fontsize_label, fontsize_title, fontsize_legend : int
        Font sizes

    Returns
    -------
    fig : matplotlib Figure
    """
    # Get data organized by concentration
    pressures, sigmas, errors, conc_labels = organize_for_pressure_plot(data, compound)

    # Note: No McCleskey model for pressure dependence currently
    # If model data is added in future, it would go here

    if compound_latex is None:
        compound_latex = compound

    fig = plot_sigma_vs_temperature(
        temp_data=pressures,  # Using pressure on x-axis instead of temperature
        sigma_data=sigmas,
        sigma_errors=errors,
        conc_labels=conc_labels,
        model_data=None,  # No pressure-dependent model yet
        xlabel=r'Pressure $P$ (MPa)',
        title=f'{compound_latex} Conductivity vs Pressure',
        show_delta=show_delta,
        show_title=show_title,
        out_file=output_file,
        colormap=colormap,
        fontsize_label=fontsize_label,
        fontsize_title=fontsize_title,
        fontsize_legend=fontsize_legend
    )

    return fig
