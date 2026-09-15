#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot Gamry impedance spectroscopy data for Cortes datasets.

Quick visualization of HiPOZ analysis results from zAnalysis CSV files.
Generates σ vs P, σ vs T, and σ vs concentration plots as PDFs.

Usage:
    python gamry_HiPOZ.py --plot-all 20250813Cortes 20250814Cortes

Or run this script directly:
    python plot_gamry_cortes.py 20250813Cortes 20250814Cortes

Requirements:
    - pandas, numpy, matplotlib (same environment as gamry_HiPOZ.py)
    - LaTeX for publication-quality rendering (optional)

For full publication plots combining benchtop + Gamry data, see:
    cortes2026/cortes2026_plots.py (follows Mahboub2026 pattern)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF output
import matplotlib.pyplot as plt

# Configure LaTeX rendering (with fallback)
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}\usepackage[version=4]{mhchem}'
    plt.rcParams['font.family'] = 'serif'
    LATEX = True
except:
    plt.rcParams['text.usetex'] = False
    LATEX = False
    print("LaTeX not available - using default rendering\n")

# ========================================
# Configuration
# ========================================

OUTPUT_DIR = 'cortes_plots'
FIGSIZE = (10, 7)
DPI = 300
MARKER_SIZE = 7
CAPSIZE = 4

# ========================================
# Functions
# ========================================

def load_gamry_data(data_dirs):
    """Load Gamry data from zAnalysis CSV files."""
    all_data = []

    for data_dir in data_dirs:
        dir_name = os.path.basename(data_dir)
        # Extract just the date portion (e.g., '20250813' from '20250813Cortes')
        date_only = dir_name[:8] if len(dir_name) >= 8 else dir_name
        zanalysis_file = os.path.join('data', data_dir, f'zAnalysis{date_only}.csv')

        if not os.path.exists(zanalysis_file):
            print(f"⚠️  File not found: {zanalysis_file}")
            continue

        print(f"Loading {zanalysis_file}...")
        df = pd.read_csv(zanalysis_file)

        # Filter to measurements with conductivity data
        df_meas = df[(df['type'] == 'measurement') & (df['conductivity_Sm'].notna())].copy()

        if len(df_meas) == 0:
            print(f"   No valid measurements found")
            continue

        print(f"   {len(df_meas)} measurements loaded")
        all_data.append(df_meas)

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} measurements\n")
    return combined

def plot_sigma_vs_pressure(data, output_file):
    """Generate σ vs P plot grouped by compound and concentration."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Group by compound and concentration
    groups = data.groupby(['comp', 'w_molal'], dropna=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for (i, ((comp, molal), group)) in enumerate(groups):
        group = group.sort_values('P_MPa')

        # Format label - handle mixtures with multiple concentrations
        if pd.notna(comp):
            if ',' in str(comp):
                # Mixture: "NaCl,MgSO4" with molal "1.5,0.4" → "NaCl+MgSO4 (1.5M, 0.4M)"
                comps = str(comp).split(',')
                molals = str(molal).split(',') if ',' in str(molal) else [str(molal)]
                comp_display = '+'.join(comps)
                molal_display = ', '.join([f"{m}M" for m in molals])
                label = f"{comp_display} ({molal_display})"
            else:
                label = f"{comp} {float(molal):.2f}M" if pd.notna(molal) else comp
        else:
            label = "Unknown"

        # Plot with error bars
        S_unc = group['conductivity_Sm'] * group['S_unc_pct'] if 'S_unc_pct' in group.columns else None

        ax.errorbar(group['P_MPa'], group['conductivity_Sm'], yerr=S_unc,
                   fmt='o-', color=colors[i], markersize=MARKER_SIZE,
                   capsize=CAPSIZE, label=label, alpha=0.7)

    ax.set_xlabel('Pressure (MPa)', fontsize=14)
    ax.set_ylabel(r'Conductivity $\sigma$ (S/m)' if LATEX else 'Conductivity σ (S/m)', fontsize=14)
    ax.set_title('Conductivity vs Pressure', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_sigma_vs_temperature(data, output_file):
    """Generate σ vs T plot grouped by compound and concentration."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    groups = data.groupby(['comp', 'w_molal'], dropna=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for (i, ((comp, molal), group)) in enumerate(groups):
        group = group.sort_values('T_K')

        # Format label - handle mixtures with multiple concentrations
        if pd.notna(comp):
            if ',' in str(comp):
                # Mixture: "NaCl,MgSO4" with molal "1.5,0.4" → "NaCl+MgSO4 (1.5M, 0.4M)"
                comps = str(comp).split(',')
                molals = str(molal).split(',') if ',' in str(molal) else [str(molal)]
                comp_display = '+'.join(comps)
                molal_display = ', '.join([f"{m}M" for m in molals])
                label = f"{comp_display} ({molal_display})"
            else:
                label = f"{comp} {float(molal):.2f}M" if pd.notna(molal) else comp
        else:
            label = "Unknown"

        S_unc = group['conductivity_Sm'] * group['S_unc_pct'] if 'S_unc_pct' in group.columns else None

        ax.errorbar(group['T_K'], group['conductivity_Sm'], yerr=S_unc,
                   fmt='o-', color=colors[i], markersize=MARKER_SIZE,
                   capsize=CAPSIZE, label=label, alpha=0.7)

    ax.set_xlabel('Temperature (K)', fontsize=14)
    ax.set_ylabel(r'Conductivity $\sigma$ (S/m)' if LATEX else 'Conductivity σ (S/m)', fontsize=14)
    ax.set_title('Conductivity vs Temperature', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_sigma_vs_concentration(data, output_file):
    """Generate σ vs concentration plot by compound."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    compounds = data['comp'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(compounds)))

    for i, comp in enumerate(compounds):
        comp_data = data[data['comp'] == comp]

        # Average by concentration
        conc_avg = comp_data.groupby('w_molal').agg({
            'conductivity_Sm': ['mean', 'std'],
            'S_unc_pct': 'mean'
        }).reset_index()

        conc = conc_avg['w_molal'].values
        sigma_mean = conc_avg[('conductivity_Sm', 'mean')].values
        sigma_std = conc_avg[('conductivity_Sm', 'std')].values

        # Use std if available, otherwise uncertainty percentage
        sigma_unc = np.where(pd.notna(sigma_std) & (sigma_std > 0),
                            sigma_std,
                            sigma_mean * conc_avg[('S_unc_pct', 'mean')].values)

        # Format compound label for mixtures
        if ',' in comp:
            label = '+'.join(comp.split(','))
        else:
            label = comp

        ax.errorbar(conc, sigma_mean, yerr=sigma_unc,
                   fmt='o-', color=colors[i], markersize=MARKER_SIZE,
                   capsize=CAPSIZE, label=label, alpha=0.7)

    ax.set_xlabel('Concentration (mol/kg)', fontsize=14)
    ax.set_ylabel(r'Conductivity $\sigma$ (S/m)' if LATEX else 'Conductivity σ (S/m)', fontsize=14)
    ax.set_title('Conductivity vs Concentration', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()
    print(f"✓ Saved: {output_file}")

# ========================================
# Main
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot Gamry impedance data from Cortes datasets',
        epilog='Example: python plot_gamry_cortes.py 20250813Cortes 20250814Cortes'
    )
    parser.add_argument('dates', nargs='*', default=['20250813Cortes', '20250814Cortes'],
                       help='Data directory names (default: 20250813Cortes 20250814Cortes)')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("Cortes Gamry Impedance Data - Conductivity Plots")
    print("=" * 70)
    print()

    # Load data
    data = load_gamry_data(args.dates)

    if data is None or len(data) == 0:
        print("ERROR: No valid data found")
        print("\nMake sure you've run HiPOZ analysis first:")
        print(f"  python gamry_HiPOZ.py --dates {' '.join(args.dates)}")
        sys.exit(1)

    # Data summary
    print("Data Summary:")
    print("-" * 70)
    print(f"Compounds: {', '.join(data['comp'].dropna().unique())}")

    # Handle mixed types in w_molal (floats for single salts, strings for mixtures like "1.5,0.4")
    molal_vals = data['w_molal'].dropna().unique()
    molal_display = []
    for m in molal_vals:
        if isinstance(m, str) and ',' in m:
            molal_display.append(f"({m})")  # Mixture
        else:
            molal_display.append(str(m))
    print(f"Concentrations (M): {', '.join(sorted(molal_display, key=lambda x: float(x.strip('()').split(',')[0])))}")

    print(f"Pressure range: {data['P_MPa'].min():.1f} - {data['P_MPa'].max():.1f} MPa")
    print(f"Temperature range: {data['T_K'].min():.1f} - {data['T_K'].max():.1f} K")
    print()

    # Generate plots
    print("Generating plots...")
    print("-" * 70)

    plot_sigma_vs_pressure(data, os.path.join(args.output, 'sigma_vs_pressure.pdf'))
    plot_sigma_vs_temperature(data, os.path.join(args.output, 'sigma_vs_temperature.pdf'))

    if data['w_molal'].nunique() > 1:
        plot_sigma_vs_concentration(data, os.path.join(args.output, 'sigma_vs_concentration.pdf'))

    print()
    print("=" * 70)
    print(f"All plots saved to: {args.output}/")
    print("=" * 70)
    print()
    print("Note: These are Gamry impedance data only.")
    print("For combined benchtop + Gamry plots, see cortes2026/cortes2026_plots.py")
    print()

if __name__ == '__main__':
    main()
