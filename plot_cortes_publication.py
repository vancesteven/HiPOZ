#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate 5 publication-quality plots for Cortes conductivity data.

Plots:
1. NaCl: σ vs P (all concentrations on one plot)
2. NaCl: σ vs T (all concentrations on one plot)
3. NaCl: σ vs concentration
4. Mixtures: σ vs T (all ratios on one plot)
5. Mixtures: σ vs concentration (all ratios on one plot)

Replicates are averaged (mean ± SEM).
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
DPI = 300
MARKER_SIZE = 8
CAPSIZE = 5
LINEWIDTH = 2


def plot_sigma_vs_pressure_combined(data_list, compound_type, output_file):
    """Plot all concentrations/ratios on one σ vs P plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    for i, (comp_key, comp, molal, subset) in enumerate(data_list):
        subset = subset.sort_values('P_MPa')

        # Format label
        if ',' in str(comp):
            comps = str(comp).split(',')
            molals = str(molal).split(',')
            label = '+'.join(comps) + f" ({', '.join([m+'M' for m in molals])})"
        else:
            label = f"{comp} {float(molal):.2f}M"

        ax.errorbar(subset['P_MPa'], subset['conductivity_Sm'],
                   yerr=subset['conductivity_sem'],
                   fmt='o', markersize=MARKER_SIZE, capsize=CAPSIZE,
                   mew=1.5, color=colors[i], label=label, alpha=0.8)

    ax.set_xlabel('Pressure (MPa)', fontsize=14)
    ax.set_ylabel(r'Conductivity $\sigma$ (S/m)' if LATEX else 'Conductivity σ (S/m)', fontsize=14)
    ax.set_title(f'{compound_type} - Conductivity vs Pressure', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()


def plot_sigma_vs_temperature_combined(data_list, compound_type, output_file):
    """Plot all concentrations/ratios on one σ vs T plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    for i, (comp_key, comp, molal, subset) in enumerate(data_list):
        subset = subset.sort_values('T_K')

        # Format label
        if ',' in str(comp):
            comps = str(comp).split(',')
            molals = str(molal).split(',')
            label = '+'.join(comps) + f" ({', '.join([m+'M' for m in molals])})"
        else:
            label = f"{comp} {float(molal):.2f}M"

        ax.errorbar(subset['T_K'], subset['conductivity_Sm'],
                   yerr=subset['conductivity_sem'],
                   fmt='o', markersize=MARKER_SIZE, capsize=CAPSIZE,
                   mew=1.5, color=colors[i], label=label, alpha=0.8)

    ax.set_xlabel('Temperature (K)', fontsize=14)
    ax.set_ylabel(r'Conductivity $\sigma$ (S/m)' if LATEX else 'Conductivity σ (S/m)', fontsize=14)
    ax.set_title(f'{compound_type} - Conductivity vs Temperature', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()


def plot_sigma_vs_concentration(data_list, compound_type, output_file):
    """Plot σ vs concentration (average across P and T)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    concentrations = []
    conductivities = []
    sems = []
    labels = []
    plot_colors = []

    for i, (comp_key, comp, molal, subset) in enumerate(data_list):
        # For mixtures with comma-separated molalities, use first concentration
        if ',' in str(molal):
            molals_split = str(molal).split(',')
            conc = float(molals_split[0])  # Use first component
            comps = str(comp).split(',')
            label = '+'.join(comps) + f" ({', '.join([m+'M' for m in molals_split])})"
        else:
            conc = float(molal)
            label = f"{comp} {conc:.2f}M"

        # Average across all P and T
        mean_sigma = subset['conductivity_Sm'].mean()
        sem_sigma = subset['conductivity_Sm'].std() / np.sqrt(len(subset))

        concentrations.append(conc)
        conductivities.append(mean_sigma)
        sems.append(sem_sigma)
        labels.append(label)
        plot_colors.append(colors[i])

    # Plot all points
    for conc, sigma, sem, label, color in zip(concentrations, conductivities, sems, labels, plot_colors):
        ax.errorbar([conc], [sigma], yerr=[sem],
                   fmt='o', markersize=MARKER_SIZE, capsize=CAPSIZE,
                   color=color, label=label, alpha=0.8)

    ax.set_xlabel('Concentration (mol/kg)', fontsize=14)
    ax.set_ylabel(r'Conductivity $\sigma$ (S/m)' if LATEX else 'Conductivity σ (S/m)', fontsize=14)
    ax.set_title(f'{compound_type} - Conductivity vs Concentration', fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate 5 publication plots for Cortes data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('dates', nargs='*', default=['20250813Cortes', '20250814Cortes'],
                       help='Data directories to process')
    parser.add_argument('--output-dir', '-o', default='cortes_plots',
                       help='Output directory')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    single_salts_dir = output_dir / 'single_salts'
    mixtures_dir = output_dir / 'mixtures'

    single_salts_dir.mkdir(parents=True, exist_ok=True)
    mixtures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cortes Publication Plots - 5 Plots Total")
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

    # Generate 5 plots
    print("Generating 5 publication plots...")
    print("-" * 70)

    plot_count = 0

    # Plot 1: NaCl σ vs P (all concentrations)
    if separated['single_salts']:
        output_p = single_salts_dir / 'NaCl_vs_pressure.pdf'
        plot_sigma_vs_pressure_combined(separated['single_salts'], 'NaCl', output_p)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_p}")

    # Plot 2: NaCl σ vs T (all concentrations)
    if separated['single_salts']:
        output_t = single_salts_dir / 'NaCl_vs_temperature.pdf'
        plot_sigma_vs_temperature_combined(separated['single_salts'], 'NaCl', output_t)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_t}")

    # Plot 3: NaCl σ vs concentration
    if separated['single_salts']:
        output_conc = single_salts_dir / 'NaCl_vs_concentration.pdf'
        plot_sigma_vs_concentration(separated['single_salts'], 'NaCl', output_conc)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_conc}")

    # Plot 4: Mixtures σ vs T (all ratios)
    if separated['mixtures']:
        output_mix_t = mixtures_dir / 'Mixtures_vs_temperature.pdf'
        plot_sigma_vs_temperature_combined(separated['mixtures'], 'NaCl+MgSO4 Mixtures', output_mix_t)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_mix_t}")

    # Plot 5: Mixtures σ vs concentration
    if separated['mixtures']:
        output_mix_conc = mixtures_dir / 'Mixtures_vs_concentration.pdf'
        plot_sigma_vs_concentration(separated['mixtures'], 'NaCl+MgSO4 Mixtures', output_mix_conc)
        plot_count += 1
        print(f"  ✓ Plot {plot_count}: {output_mix_conc}")

    print()
    print("=" * 70)
    print(f"Complete! Generated {plot_count} plots")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}/")
    print("  single_salts/")
    print("    - NaCl_vs_pressure.pdf")
    print("    - NaCl_vs_temperature.pdf")
    print("    - NaCl_vs_concentration.pdf")
    print("  mixtures/")
    print("    - Mixtures_vs_temperature.pdf")
    print("    - Mixtures_vs_concentration.pdf")
    print()


if __name__ == '__main__':
    main()
