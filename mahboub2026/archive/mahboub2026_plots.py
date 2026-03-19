#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication plots for Mahboub et al. (2026, in press).

This script generates conductivity vs concentration and conductivity vs temperature
plots for benchtop probe measurements integrated with Gamry impedance data.

Data sources:
1. Benchtop probe data: Mahboub2026BenchtopData.csv
2. Gamry impedance data: data/20250813Mahboub2026
3. McCleskey et al. (2011) reference data

Outputs:
- 10 plots (5 compounds × 2 plot types) in mahboub_plots/ directory
- Automatic overlay of Gamry measurements on benchtop data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generalized study plotting functions
from study_plots import (
    load_study_data,
    plot_study_concentration,
    plot_study_temperature,
    organize_for_conc_plot
)

# Plot configuration
from config_plots import (
    FONTSIZE_AXIS_LABEL,
    FONTSIZE_TITLE,
    FONTSIZE_LEGEND,
    COLORMAP_CONCENTRATION,
    COLORMAP_TEMPERATURE,
    SHOW_LEGEND,
    SHOW_TITLE,
    SHOW_DELTA
)

# ========================================
# Configuration
# ========================================

GAMRY_DATA_DIR = 'data/20250813Mahboub2026'
OUTPUT_DIR = 'mahboub_plots'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# Load Data
# ========================================

print("=" * 70)
print("Mahboub et al. (2026) - Conductivity Plots")
print("=" * 70)
print()

# Load benchtop data from CSV
print("Loading benchtop data from Mahboub2026BenchtopData.csv...")
benchtop_data = load_study_data('Mahboub2026BenchtopData.csv')
print(f"  Found {len(benchtop_data)} compounds")
print()

# Load Gamry data if available
gamry_df = None
if os.path.exists(GAMRY_DATA_DIR):
    curated_file = os.path.join(GAMRY_DATA_DIR, 'hipoz_latest_curated.csv')
    if os.path.exists(curated_file):
        print(f"Loading Gamry data from {curated_file}...")
        gamry_df = pd.read_csv(curated_file)
        print(f"  Found {len(gamry_df)} Gamry measurements")
        print()
    else:
        print(f"Note: Gamry curated file not found at {curated_file}")
        print("      Run GUI analysis first to generate curated data")
        print()

# ========================================
# Plot 1: NaCl σ vs Concentration (with Delta subplot)
# ========================================

print("Generating Plot 1: NaCl Conductivity vs Concentration...")

# Get Gamry overlay data (source='Gamry20250813')
gamry_nacl_conc, gamry_nacl_sigma, gamry_nacl_errors, _ = \
    organize_for_conc_plot(benchtop_data, 'NaCl', source_filter='Gamry20250813')

# Prepare Gamry data tuple (if available)
gamry_data_nacl = None
if len(gamry_nacl_sigma) > 0 and len(gamry_nacl_sigma[0]) > 0:
    gamry_data_nacl = (
        gamry_nacl_conc,
        gamry_nacl_sigma[0],  # First temp (20°C = 293.15 K)
        gamry_nacl_errors[0],
        'Gamry (293.15 K)'
    )

fig = plot_study_concentration(
    data=benchtop_data,
    compound='NaCl',
    output_file=os.path.join(OUTPUT_DIR, 'nacl_vs_concentration.pdf'),
    gamry_data=gamry_data_nacl,
    show_delta=True,
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/nacl_vs_concentration.pdf")
if gamry_data_nacl:
    print(f"  Overlay: {len(gamry_data_nacl[0])} Gamry points added")
print()

# ========================================
# Plot 2: NaCl σ vs Temperature (with Delta subplot)
# ========================================

print("Generating Plot 2: NaCl Conductivity vs Temperature...")

fig = plot_study_temperature(
    data=benchtop_data,
    compound='NaCl',
    output_file=os.path.join(OUTPUT_DIR, 'nacl_vs_temperature.pdf'),
    show_delta=True,
    colormap=COLORMAP_TEMPERATURE,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/nacl_vs_temperature.pdf")
print()

# ========================================
# Plot 3: MgSO4 σ vs Concentration (with Delta subplot)
# ========================================

print("Generating Plot 3: MgSO4 Conductivity vs Concentration...")

# Get Gamry overlay data
gamry_mgso4_conc, gamry_mgso4_sigma, gamry_mgso4_errors, _ = \
    organize_for_conc_plot(benchtop_data, 'MgSO4', source_filter='GamryLiterature')

gamry_data_mgso4 = None
if len(gamry_mgso4_sigma) > 0 and len(gamry_mgso4_sigma[0]) > 0:
    gamry_data_mgso4 = (
        gamry_mgso4_conc,
        gamry_mgso4_sigma[0],
        gamry_mgso4_errors[0],
        'Gamry (292.65 K)'
    )

fig = plot_study_concentration(
    data=benchtop_data,
    compound='MgSO4',
    output_file=os.path.join(OUTPUT_DIR, 'mgso4_vs_concentration.pdf'),
    gamry_data=gamry_data_mgso4,
    show_delta=True,
    compound_latex=r'MgSO$_4$',
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/mgso4_vs_concentration.pdf")
if gamry_data_mgso4:
    print(f"  Overlay: {len(gamry_data_mgso4[0])} Gamry points added")
print()

# ========================================
# Plot 4: MgSO4 σ vs Temperature (with Delta subplot)
# ========================================

print("Generating Plot 4: MgSO4 Conductivity vs Temperature...")

fig = plot_study_temperature(
    data=benchtop_data,
    compound='MgSO4',
    output_file=os.path.join(OUTPUT_DIR, 'mgso4_vs_temperature.pdf'),
    show_delta=True,
    compound_latex=r'MgSO$_4$',
    colormap=COLORMAP_TEMPERATURE,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/mgso4_vs_temperature.pdf")
print()

# ========================================
# Plot 5: NH4Cl σ vs Concentration (with Delta subplot)
# ========================================

print("Generating Plot 5: NH4Cl Conductivity vs Concentration...")

fig = plot_study_concentration(
    data=benchtop_data,
    compound='NH4Cl',
    output_file=os.path.join(OUTPUT_DIR, 'nh4cl_vs_concentration.pdf'),
    show_delta=True,
    compound_latex=r'NH$_4$Cl',
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/nh4cl_vs_concentration.pdf")
print()

# ========================================
# Plot 6: NH4Cl σ vs Temperature (with Delta subplot)
# ========================================

print("Generating Plot 6: NH4Cl Conductivity vs Temperature...")

fig = plot_study_temperature(
    data=benchtop_data,
    compound='NH4Cl',
    output_file=os.path.join(OUTPUT_DIR, 'nh4cl_vs_temperature.pdf'),
    show_delta=True,
    compound_latex=r'NH$_4$Cl',
    colormap=COLORMAP_TEMPERATURE,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/nh4cl_vs_temperature.pdf")
print()

# ========================================
# Plot 7: Na2CO3 σ vs Concentration (with Delta subplot)
# ========================================

print("Generating Plot 7: Na2CO3 Conductivity vs Concentration...")

fig = plot_study_concentration(
    data=benchtop_data,
    compound='Na2CO3',
    output_file=os.path.join(OUTPUT_DIR, 'na2co3_vs_concentration.pdf'),
    show_delta=True,
    compound_latex=r'Na$_2$CO$_3$',
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/na2co3_vs_concentration.pdf")
print()

# ========================================
# Plot 8: Na2CO3 σ vs Temperature (with Delta subplot)
# ========================================

print("Generating Plot 8: Na2CO3 Conductivity vs Temperature...")

fig = plot_study_temperature(
    data=benchtop_data,
    compound='Na2CO3',
    output_file=os.path.join(OUTPUT_DIR, 'na2co3_vs_temperature.pdf'),
    show_delta=True,
    compound_latex=r'Na$_2$CO$_3$',
    colormap=COLORMAP_TEMPERATURE,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/na2co3_vs_temperature.pdf")
print()

# ========================================
# Plot 9: Mixture σ vs Concentration
# ========================================

print("Generating Plot 9: Mixture (1:1:1) Conductivity vs Concentration...")

# Mixture has custom ion stoichiometry - skip Delta subplot for now
fig = plot_study_concentration(
    data=benchtop_data,
    compound='Mixture',
    output_file=os.path.join(OUTPUT_DIR, 'mixture_vs_concentration.pdf'),
    show_delta=False,  # No McCleskey model for mixtures yet
    compound_latex=r'MgSO$_4$:NaCl:Na$_2$CO$_3$ (1:1:1)',
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/mixture_vs_concentration.pdf")
print()

# ========================================
# Plot 10: Mixture σ vs Temperature (no Delta)
# ========================================

print("Generating Plot 10: Mixture (1:1:1) Conductivity vs Temperature...")

fig = plot_study_temperature(
    data=benchtop_data,
    compound='Mixture',
    output_file=os.path.join(OUTPUT_DIR, 'mixture_vs_temperature.pdf'),
    show_delta=False,  # No McCleskey model for mixtures yet
    compound_latex=r'MgSO$_4$:NaCl:Na$_2$CO$_3$ (1:1:1)',
    colormap=COLORMAP_TEMPERATURE,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/mixture_vs_temperature.pdf")
print()

# ========================================
# Summary Statistics
# ========================================

print("=" * 70)
print("Summary Statistics")
print("=" * 70)
print()

for compound in ['NaCl', 'MgSO4', 'NH4Cl', 'Na2CO3', 'Mixture']:
    comp_data = benchtop_data[compound]
    benchtop_mask = comp_data['source'] == 'benchtop'

    n_measurements = np.sum(benchtop_mask)
    n_replicates = len(np.unique(comp_data['replicates'][benchtop_mask]))
    n_temps = len(np.unique(comp_data['temperatures_K'][benchtop_mask]))
    n_concs = len(np.unique(comp_data['concentration_molal'][benchtop_mask]))

    print(f"{compound}:")
    print(f"  Total measurements: {n_measurements}")
    print(f"  Replicates per condition: {n_replicates}")
    print(f"  Temperatures: {n_temps}")
    print(f"  Concentrations: {n_concs}")
    print()

# Gamry data summary
if gamry_df is not None:
    print("Gamry Impedance Data:")
    print(f"  Total measurements: {len(gamry_df)}")
    if 'Comp' in gamry_df.columns:
        for comp in gamry_df['Comp'].dropna().unique():
            n_comp = len(gamry_df[gamry_df['Comp'] == comp])
            print(f"  {comp}: {n_comp} measurements")
    print()

print("=" * 70)
print("All plots saved to:", OUTPUT_DIR)
print("=" * 70)
print()

print("Publication-ready plots:")
print("  - Benchtop probe data: Mahboub2026BenchtopData.csv")
print("  - Gamry impedance overlay: data/20250813Mahboub2026")
print("  - All figures: mahboub_plots/*_vs_*.pdf")
print()
print("Citation:")
print("  Mahboub, R., et al. (2026)")
print()
print("Key features:")
print("  - Data stored in easy-to-edit CSV format")
print("  - Automatic integration with Gamry impedance measurements")
print("  - Reusable plotting functions (plotting.py)")
print("  - Command-line scriptable for reproducibility")
