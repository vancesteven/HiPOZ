#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication plots for Cortes data with integrated McCleskey comparison.

This script generates:
- Standard plots (all pressure data)
- McCleskey-integrated plots (low-P data with Delta subplots, like Mahboub)

Usage:
    # Basic plots (no McCleskey)
    python plot_cortes_with_mccleskey.py

    # With integrated McCleskey comparison (low-P data only)
    python plot_cortes_with_mccleskey.py --show-mccleskey

    # Custom pressure threshold
    python plot_cortes_with_mccleskey.py --show-mccleskey --p-threshold 3.0
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import data processing utilities
import cortes_data_processing as cdp
import cortes_mccleskey as cm

# Import McCleskey model computation and utilities
from study_plots import compute_mccleskey_model, ION_SPECS
from plotting import pdiff

# Configure LaTeX
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage[version=4]{mhchem}'
    plt.rcParams['font.family'] = 'serif'
    LATEX = True
except:
    plt.rcParams['text.usetex'] = False
    LATEX = False

# Plot settings
FIGSIZE = (10, 7)
FIGSIZE_DELTA = (10, 10)  # Taller for Delta subplot
DPI = 300
MARKER_SIZE = 8
CAPSIZE = 5
LINEWIDTH = 2


def plot_with_mccleskey_subplot(subset, compound, x_data, x_label,
                                output_file, model_key='model_Sm',
                                delta_key='delta_pct'):
    """
    Plot data with McCleskey model and Delta subplot.

    Parameters
    ----------
    subset : pandas.DataFrame
        Data subset for this compound
    compound : str
        Compound name
    x_data : str
        Column name for x-axis ('P_MPa', 'T_K', or 'w_molal')
    x_label : str
        X-axis label
    output_file : Path
        Output file path
    model_key : str
        Column name for model data
    delta_key : str
        Column name for delta values
    """
    # Check if model data exists
    if model_key not in subset.columns or subset[model_key].isna().all():
        print(f"  No McCleskey model data for {compound}, skipping...")
        return

    # Ensure numeric types for all numeric columns
    subset = subset.copy()
    subset['w_molal'] = pd.to_numeric(subset['w_molal'], errors='coerce')
    subset['T_K'] = pd.to_numeric(subset['T_K'], errors='coerce')
    subset['P_MPa'] = pd.to_numeric(subset['P_MPa'], errors='coerce')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_DELTA,
                                     height_ratios=[2, 1], sharex=True)

    # Sort by x-axis
    subset = subset.sort_values(x_data)

    # Colors for different concentrations/temperatures
    cmap = plt.cm.tab10
    unique_groups = subset['w_molal'].unique() if x_data != 'w_molal' else subset['T_K'].unique()
    colors = cmap(np.arange(len(unique_groups)) % 10)

    # Plot data and model
    for i, group_val in enumerate(unique_groups):
        if x_data == 'w_molal':
            group_mask = np.abs(subset['T_K'] - group_val) < 0.5
            label = f"{group_val:.1f} K"
        else:
            group_mask = np.abs(subset['w_molal'] - group_val) < 0.001
            label = f"{group_val:.2f} mol/kg"

        group_data = subset[group_mask]

        if len(group_data) == 0:
            continue

        # Experimental data
        ax1.errorbar(group_data[x_data], group_data['conductivity_Sm'],
                    yerr=group_data.get('conductivity_sem', None),
                    fmt='o', color=colors[i], markersize=MARKER_SIZE,
                    capsize=CAPSIZE, mew=1.5, label=label, zorder=3)

        # Model data (dashed line)
        if model_key in group_data.columns:
            valid = group_data[model_key].notna()
            if valid.any():
                ax1.plot(group_data[x_data][valid], group_data[model_key][valid],
                        ls='--', color=colors[i], lw=LINEWIDTH, alpha=0.7, zorder=1)

        # Delta subplot
        if delta_key in group_data.columns:
            valid = group_data[delta_key].notna()
            if valid.any():
                ax2.errorbar(group_data[x_data][valid], group_data[delta_key][valid],
                            fmt='o', color=colors[i], markersize=MARKER_SIZE-2,
                            mew=1.2, capsize=CAPSIZE-2, zorder=3)

    # Formatting - Top panel
    ax1.set_ylabel(r'$\sigma$ (S/m)', fontsize=14)
    ax1.set_title(f'{compound} Conductivity with McCleskey Model', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')

    # Add composition limit if available
    limit = cm.get_mccleskey_limit(compound)
    if limit is not None and x_data == 'w_molal':
        ax1.axvline(limit, ls='--', color=(0.8, 0, 0), lw=2.5, alpha=0.8, zorder=2.5)
        y_pos = ax1.get_ylim()[0] + 0.95 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        ax1.text(limit * 1.01, y_pos, f"McCleskey limit\n({limit:.4f} mol/kg)",
                color=(0.8, 0, 0), fontsize=10, fontweight="bold",
                ha="left", va="top", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", edgecolor=(0.8, 0, 0), alpha=0.8))

    # Formatting - Delta panel
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel(r'$\Delta$ (\%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', ls='-', lw=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()
    print(f"  ✓ Saved: {output_file}")


def plot_with_integrated_mccleskey(data_list, compound, x_data, x_label, y_data,
                                   output_file, p_threshold=5.0,
                                   use_pressure_colormap=False):
    """
    Plot with integrated McCleskey predictions and delta subplot (Mahboub-style).

    This creates a 2-panel plot:
    - Top: experimental data (markers) + McCleskey model (dashed lines)
    - Bottom: Delta subplot showing percent deviation

    Parameters
    ----------
    data_list : list of tuples
        List of (comp_key, comp, molal, subset) tuples
    compound : str
        Compound name (e.g., 'NaCl', 'MgSO4')
    x_data : str
        Column for x-axis ('T_K' or 'w_molal')
    x_label : str
        X-axis label
    y_data : str
        Column for y-axis ('conductivity_Sm')
    output_file : Path
        Output file path
    p_threshold : float
        Pressure threshold for McCleskey validity (MPa)
    use_pressure_colormap : bool
        If True, use pressure colormap (only for temperature plots)
    """
    # Check if this compound has McCleskey model
    if compound not in ION_SPECS:
        print(f"  No McCleskey model for {compound}, skipping...")
        return

    # Combine all data and filter to low pressure
    all_data = []
    for comp_key, comp, molal, subset in data_list:
        subset = subset.copy()
        # Ensure numeric types
        subset['w_molal'] = pd.to_numeric(subset['w_molal'], errors='coerce')
        subset['T_K'] = pd.to_numeric(subset['T_K'], errors='coerce')
        subset['P_MPa'] = pd.to_numeric(subset['P_MPa'], errors='coerce')
        subset['conductivity_Sm'] = pd.to_numeric(subset['conductivity_Sm'], errors='coerce')
        all_data.append(subset)

    if len(all_data) == 0:
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Filter to low pressure for McCleskey comparison
    low_p = combined[combined['P_MPa'] <= p_threshold].copy()

    # Remove rows with NaN concentrations (bad data)
    low_p = low_p[low_p['w_molal'].notna()]

    if len(low_p) == 0:
        print(f"  No low-pressure data (P ≤ {p_threshold} MPa) for {compound}")
        return

    print(f"  Filtered to {len(low_p)} points with P ≤ {p_threshold} MPa")

    # Create figure with two subplots (top: data+model, bottom: delta)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                     height_ratios=[2, 1], sharex=True)

    # Sort by x-axis
    low_p = low_p.sort_values(x_data)

    # Determine grouping variable (opposite of x-axis)
    if x_data == 'w_molal':
        # Plotting vs concentration, so group by temperature
        group_col = 'T_K'
        group_label_fmt = lambda v: f"{v:.1f} K"
    else:
        # Plotting vs temperature, so group by concentration
        group_col = 'w_molal'
        group_label_fmt = lambda v: f"{v:.2f} mol/kg"

    # Get unique groups
    unique_groups = np.sort(low_p[group_col].unique())

    # Colors for different groups
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(unique_groups)) % 10)

    # For concentration plots: compute single McCleskey model at average temperature
    if x_data == 'w_molal':
        avg_temp = low_p['T_K'].mean()
        print(f"    Computing unified McCleskey model at T_avg = {avg_temp:.1f} K")

        # Get concentration range from all data
        w_min = low_p['w_molal'].min()
        w_max = low_p['w_molal'].max()
        w_range = w_max - w_min

        # Expand slightly if needed
        if w_range < 0.01:
            w_range = 0.2
            w_min = low_p['w_molal'].mean() - w_range / 2
            w_max = low_p['w_molal'].mean() + w_range / 2

        # Generate 50 uniform concentration points
        w_model = np.linspace(w_min, w_max, 50)

        # Compute McCleskey at average temperature
        concs_model = w_model
        temps_model = np.array([avg_temp])
        model_result = compute_mccleskey_model(concs_model, temps_model, compound=compound)
        model_sigma_unified = model_result[0]

        # Plot single unified model line (black dashed)
        ax1.plot(w_model, model_sigma_unified, ls='--', color='black',
                lw=LINEWIDTH, alpha=0.7, zorder=1, label=f'McCleskey ({avg_temp:.1f} K)')

    # For each group, plot data and compute delta
    for i, group_val in enumerate(unique_groups):
        # Get data for this group
        group_mask = np.abs(low_p[group_col] - group_val) < 0.01
        group_data = low_p[group_mask].copy()

        if len(group_data) == 0:
            continue

        label = group_label_fmt(group_val)

        # Get x and y values
        x_vals = group_data[x_data].values
        y_vals = group_data[y_data].values

        # Use 5% uncertainty for Gamry data (user-specified)
        # If SEM is available, use larger of SEM or 5%
        y_sem = group_data.get('conductivity_sem', pd.Series([None]*len(group_data))).values
        y_err_5pct = 0.05 * y_vals  # 5% uncertainty

        # Use whichever is larger (or 5% if SEM is NaN)
        y_err = np.where(np.isnan(y_sem) | (y_err_5pct > y_sem), y_err_5pct, y_sem)

        # Plot experimental data
        ax1.errorbar(x_vals, y_vals, yerr=y_err,
                    fmt='o', color=colors[i], markersize=MARKER_SIZE,
                    capsize=CAPSIZE, mew=1.5, label=label, zorder=3)

        # Compute McCleskey model for this group
        try:
            if x_data == 'w_molal':
                # Concentration plot: unified model already plotted
                # Also plot individual line segments for each temperature

                # Create uniform concentration grid for smooth line segments
                x_min, x_max = x_vals.min(), x_vals.max()
                x_range = x_max - x_min

                # For single point, expand slightly around it
                if len(x_vals) == 1 or x_range < 1e-6:
                    x_range = 0.2  # mol/kg
                    x_min = x_vals[0] - x_range / 2
                    x_max = x_vals[0] + x_range / 2

                # Generate 50 uniform points for smooth line segment
                x_model = np.linspace(x_min, x_max, 50)

                # For fixed T, varying w: pass all concentrations, single temperature
                concs_model = x_model
                temps_model = np.array([group_val])  # Single temperature
                concs_data = x_vals
                temps_data = np.array([group_val])

                # Compute model: returns list of length 1, containing array of length n_concs
                model_result = compute_mccleskey_model(concs_model, temps_model, compound=compound)
                model_sigma = model_result[0]  # Extract the single array

                # Compute at data points for delta
                model_result_data = compute_mccleskey_model(concs_data, temps_data, compound=compound)
                model_sigma_data = model_result_data[0]

                # Plot individual temperature line segment (colored dashed line)
                ax1.plot(x_model, model_sigma, ls='--', color=colors[i],
                        lw=LINEWIDTH, alpha=0.7, zorder=2)

            else:
                # Temperature plot: plot individual model line for each concentration
                # Create uniform grid for smooth model lines
                x_min, x_max = x_vals.min(), x_vals.max()
                x_range = x_max - x_min

                # For single point, expand slightly around it
                if len(x_vals) == 1 or x_range < 1e-6:
                    x_range = 2.0  # ±1 K
                    x_min = x_vals[0] - x_range / 2
                    x_max = x_vals[0] + x_range / 2

                # Generate 50 uniform points for smooth line
                x_model = np.linspace(x_min, x_max, 50)

                # For fixed w, varying T: pass single concentration, all temperatures
                concs_model = np.array([group_val])  # Single concentration
                temps_model = x_model
                concs_data = np.array([group_val])
                temps_data = x_vals

                # Compute model: returns list of length n_temps, each containing array of length 1
                model_result = compute_mccleskey_model(concs_model, temps_model, compound=compound)
                # Extract first element of each array (since we only have one concentration)
                model_sigma = np.array([r[0] for r in model_result])

                # Compute at data points for delta
                model_result_data = compute_mccleskey_model(concs_data, temps_data, compound=compound)
                model_sigma_data = np.array([r[0] for r in model_result_data])

                # Plot model as smooth dashed line
                ax1.plot(x_model, model_sigma, ls='--', color=colors[i],
                        lw=LINEWIDTH, alpha=0.7, zorder=1)

            # Compute delta (percent difference) at data points
            delta = pdiff(y_vals, model_sigma_data)

            # Propagate uncertainty to delta
            # δ(Δ) = |100 / σ_model| * δ(σ_exp) = 100 * σ_exp / σ_model
            delta_err = 100.0 * y_err / model_sigma_data

            # Plot delta in bottom panel with error bars
            ax2.errorbar(x_vals, delta, yerr=delta_err, fmt='o', color=colors[i],
                        markersize=MARKER_SIZE-2, mew=1.2, capsize=CAPSIZE-2,
                        zorder=3)

        except Exception as e:
            print(f"  Warning: Could not compute McCleskey model for {label}: {e}")
            continue

    # Formatting - Top panel
    ax1.set_ylabel(r'$\sigma$ (S/m)', fontsize=14)
    ax1.set_title(f'{compound} Conductivity with McCleskey Model', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')

    # Set y-axis minimum to 0 (conductivity cannot be negative)
    ax1.set_ylim(bottom=0)

    # Add composition limit if plotting vs concentration
    if x_data == 'w_molal':
        limit = cm.get_mccleskey_limit(compound)
        if limit is not None:
            ax1.axvline(limit, ls='--', color=(0.8, 0, 0), lw=2.5, alpha=0.8, zorder=2.5)
            y_pos = ax1.get_ylim()[0] + 0.95 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
            ax1.text(limit * 1.01, y_pos, f"McCleskey limit\n({limit:.4f} mol/kg)",
                    color=(0.8, 0, 0), fontsize=10, fontweight="bold",
                    ha="left", va="top", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", edgecolor=(0.8, 0, 0), alpha=0.8))

    # Formatting - Delta panel
    ax2.set_xlabel(x_label, fontsize=14)
    ax2.set_ylabel(r'$\Delta$ (\%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', ls='-', lw=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()
    print(f"  ✓ Saved: {output_file}")


def plot_simple(data_list, compound_type, x_data, x_label, y_data, output_file,
                use_pressure_colormap=False):
    """
    Plot without McCleskey comparison (all pressure data).

    Parameters
    ----------
    data_list : list of tuples
        List of (comp_key, comp, molal, subset) tuples
    compound_type : str
        Type label (e.g., 'NaCl', 'Mixtures')
    x_data : str
        Column for x-axis
    x_label : str
        X-axis label
    y_data : str
        Column for y-axis
    output_file : Path
        Output file path
    use_pressure_colormap : bool
        If True and x_data is 'T_K', use pressure as colormap
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Use pressure colormap for temperature plots if requested
    if use_pressure_colormap and x_data == 'T_K' and len(data_list) > 0:
        # Collect all data for colormap scaling
        all_pressures = []
        all_x = []
        all_y = []
        all_yerr = []

        for comp_key, comp, molal, subset in data_list:
            subset = subset.sort_values(x_data)
            if 'P_MPa' in subset.columns:
                all_pressures.extend(subset['P_MPa'].values)
                all_x.extend(subset[x_data].values)
                all_y.extend(subset[y_data].values)
                if 'conductivity_sem' in subset.columns:
                    all_yerr.extend(subset['conductivity_sem'].values)

        if len(all_pressures) > 0:
            # Create scatter plot with pressure colormap
            norm = plt.Normalize(vmin=min(all_pressures), vmax=max(all_pressures))
            cmap = plt.cm.viridis

            scatter = ax.scatter(all_x, all_y, c=all_pressures, cmap=cmap, norm=norm,
                               s=MARKER_SIZE**2, alpha=0.8, edgecolors='black', linewidths=1)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Pressure (MPa)')
            cbar.ax.tick_params(labelsize=10)

            # Add error bars separately (no color)
            if len(all_yerr) > 0 and any(pd.notna(all_yerr)):
                ax.errorbar(all_x, all_y, yerr=all_yerr, fmt='none',
                          ecolor='gray', capsize=CAPSIZE, alpha=0.3, zorder=1)
    else:
        # Standard categorical coloring
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

        for i, (comp_key, comp, molal, subset) in enumerate(data_list):
            subset = subset.sort_values(x_data)

            # Format label
            if ',' in str(comp):
                comps = str(comp).split(',')
                molals = str(molal).split(',')
                label = '+'.join(comps) + f" ({', '.join([m+'M' for m in molals])})"
            else:
                label = f"{comp} {float(molal):.2f}M"

            ax.errorbar(subset[x_data], subset[y_data],
                       yerr=subset.get('conductivity_sem', None),
                       fmt='o', markersize=MARKER_SIZE, capsize=CAPSIZE,
                       mew=1.5, color=colors[i], label=label, alpha=0.8)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(r'$\sigma$ (S/m)' if LATEX else 'σ (S/m)', fontsize=14)
    ax.set_title(f'{compound_type} - Conductivity vs {x_label.split()[0]}', fontsize=16)

    # Only show legend if not using pressure colormap
    if not (use_pressure_colormap and x_data == 'T_K'):
        ax.legend(fontsize=10, loc='best')

    ax.grid(True, alpha=0.3)

    # Set y-axis minimum to 0, except for pressure plots
    if x_data != 'P_MPa':
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate Cortes publication plots with optional McCleskey comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('dates', nargs='*', default=['20250813Cortes', '20250814Cortes', '20250815Cortes'],
                       help='Data directories to process')
    parser.add_argument('--output-dir', '-o', default='cortes_plots',
                       help='Output directory')
    parser.add_argument('--show-mccleskey', '-m', action='store_true',
                       help='Include McCleskey model comparison (low-P data only)')
    parser.add_argument('--p-threshold', '-p', type=float, default=5.0,
                       help='Pressure threshold for McCleskey comparison (MPa, default: 5.0)')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    single_salts_dir = output_dir / 'single_salts'
    mixtures_dir = output_dir / 'mixtures'
    mccleskey_dir = output_dir / 'mccleskey_comparison'

    single_salts_dir.mkdir(parents=True, exist_ok=True)
    mixtures_dir.mkdir(parents=True, exist_ok=True)

    if args.show_mccleskey:
        mccleskey_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cortes Publication Plots")
    if args.show_mccleskey:
        print(f"With McCleskey comparison (P ≤ {args.p_threshold} MPa)")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    data = cdp.load_cortes_data(args.dates)

    if data is None or len(data) == 0:
        print("ERROR: No valid data found")
        sys.exit(1)

    print(f"Total raw measurements: {len(data)}")
    print()

    # Average replicates
    print("Averaging replicates...")
    averaged = cdp.average_replicates(data)
    print(f"Averaged data points: {len(averaged)}")
    print(f"Average n_replicates: {averaged['n_replicates'].mean():.1f}")
    print()

    # Separate by composition
    separated = cdp.separate_by_composition(averaged)

    print(f"Single salts: {len(separated['single_salts'])} compositions")
    print(f"Mixtures: {len(separated['mixtures'])} compositions")
    print()

    # Generate standard plots (all pressure data)
    print("Generating standard plots (all pressure data)...")
    print("-" * 70)

    plot_count = 0

    # NaCl plots (filter to 0.5 molal only)
    if separated['single_salts']:
        # Filter to only 0.5 molal NaCl
        nacl_05m = [(ck, c, m, s) for ck, c, m, s in separated['single_salts']
                    if c == 'NaCl' and abs(float(m) - 0.5) < 0.01]

        if nacl_05m:
            output_p = single_salts_dir / 'NaCl_0.5M_vs_pressure.pdf'
            plot_simple(nacl_05m, 'NaCl (0.5 mol/kg)', 'P_MPa', 'Pressure (MPa)',
                       'conductivity_Sm', output_p)
            plot_count += 1
            print(f"  ✓ Plot {plot_count}: {output_p}")

            output_t = single_salts_dir / 'NaCl_0.5M_vs_temperature.pdf'
            plot_simple(nacl_05m, 'NaCl (0.5 mol/kg)', 'T_K', 'Temperature (K)',
                       'conductivity_Sm', output_t, use_pressure_colormap=True)
            plot_count += 1
            print(f"  ✓ Plot {plot_count}: {output_t}")

    # Mixture plots
    if separated['mixtures']:
        output_mix_t = mixtures_dir / 'Mixtures_vs_temperature.pdf'
        plot_simple(separated['mixtures'], 'NaCl+MgSO4 Mixtures', 'T_K',
                   'Temperature (K)', 'conductivity_Sm', output_mix_t)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_mix_t}")

        # Add mixtures vs MgSO4 concentration plot
        # Extract MgSO4 concentration from mixture data
        mixtures_with_conc = []
        for comp_key, comp, molal, subset in separated['mixtures']:
            subset = subset.copy()
            # Parse molal string like "1.5,0.4" to get MgSO4 concentration (second value)
            if ',' in str(molal):
                mgso4_conc = float(str(molal).split(',')[1])
            else:
                mgso4_conc = 0.0
            subset['MgSO4_molal'] = mgso4_conc
            mixtures_with_conc.append((comp_key, comp, molal, subset))

        output_mix_c = mixtures_dir / 'Mixtures_vs_MgSO4_concentration.pdf'
        plot_simple(mixtures_with_conc, 'NaCl+MgSO4 Mixtures', 'MgSO4_molal',
                   r'MgSO4 Concentration (mol/kg$_{\mathrm{H_2O}}$)',
                   'conductivity_Sm', output_mix_c)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_mix_c}")

    print()

    # Generate integrated McCleskey comparison plots if requested
    if args.show_mccleskey:
        print(f"Generating integrated McCleskey comparison plots (P ≤ {args.p_threshold} MPa)...")
        print("-" * 70)

        # Process single salts that have McCleskey models
        for compound in ['NaCl', 'KCl', 'Na2SO4', 'MgSO4']:
            # Get data for this compound
            comp_data = [(ck, c, m, s) for ck, c, m, s in separated['single_salts']
                        if c == compound]

            if len(comp_data) == 0:
                continue

            print(f"\n{compound}:")

            # Temperature plots (vs T for different concentrations)
            output_file = mccleskey_dir / f'{compound}_vs_temperature_mccleskey.pdf'
            plot_with_integrated_mccleskey(comp_data, compound, 'T_K', 'Temperature (K)',
                                          'conductivity_Sm', output_file, args.p_threshold)
            plot_count += 1

            # Concentration plots (vs w for different temperatures)
            # Only if we have multiple concentrations
            concentrations = set()
            for ck, c, m, s in comp_data:
                try:
                    concentrations.add(float(m))
                except:
                    pass

            if len(concentrations) > 1:
                output_file = mccleskey_dir / f'{compound}_vs_concentration_mccleskey.pdf'
                plot_with_integrated_mccleskey(comp_data, compound, 'w_molal',
                                              r'Concentration (mol/kg$_{\mathrm{H_2O}}$)',
                                              'conductivity_Sm', output_file, args.p_threshold)
                plot_count += 1

        print()

    # Summary
    print("=" * 70)
    print(f"Complete! Generated {plot_count} plots")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}/")
    print("  single_salts/ - High-pressure Gamry data")
    print("  mixtures/ - Mixture compositions")
    if args.show_mccleskey:
        print("  mccleskey_comparison/ - Low-P data with model comparison")
    print()


if __name__ == '__main__':
    main()
