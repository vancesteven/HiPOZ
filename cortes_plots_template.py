#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template for Cortes et al. conductivity study plots.

This template demonstrates how to create conductivity plots for a new study
using the generalized plotting functions from study_plots.py.

To use this template:
1. Copy this file and rename (e.g., cortes2027_plots.py)
2. Create your data CSV file (see CortesDataTemplate.csv)
3. Update OUTPUT_DIR, compounds, and Gamry data paths below
4. Run: python cortes2027_plots.py
"""

import os
import numpy as np
import pandas as pd

# Generalized study plotting functions
from study_plots import (
    load_study_data,
    plot_study_concentration,
    plot_study_temperature,
    plot_study_pressure,
    organize_for_conc_plot
)

# Import universal plot settings
from config_plots import (
    FONTSIZE_AXIS_LABEL,
    FONTSIZE_TITLE,
    FONTSIZE_LEGEND,
    COLORMAP_CONCENTRATION,
    COLORMAP_TEMPERATURE,
    COLORMAP_PRESSURE,
    SHOW_LEGEND,
    SHOW_TITLE
)

# ========================================
# Configuration
# ========================================

# Data and output directories
DATA_CSV = 'CortesData.csv'  # Your benchtop data CSV
GAMRY_DATA_DIR = 'data/cortes_gamry'  # Optional: directory with Gamry data
OUTPUT_DIR = 'cortes_plots'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# Load Data
# ========================================

print("=" * 70)
print("Cortes et al. - Conductivity Plots")
print("=" * 70)
print()

print(f"Loading benchtop data from {DATA_CSV}...")
benchtop_data = load_study_data(DATA_CSV)
print(f"  Found {len(benchtop_data)} compounds")
print()

# Optional: Load Gamry impedance data
# gamry_df = None
# if os.path.exists(GAMRY_DATA_DIR):
#     curated_file = os.path.join(GAMRY_DATA_DIR, 'hipoz_latest_curated.csv')
#     if os.path.exists(curated_file):
#         print(f"Loading Gamry data from {curated_file}...")
#         gamry_df = pd.read_csv(curated_file)
#         print(f"  Found {len(gamry_df)} Gamry measurements")
#         print()

# ========================================
# Example Plot 1: KCl σ vs Concentration
# ========================================

print("Generating Plot 1: KCl Conductivity vs Concentration...")

# Optional: Get Gamry overlay data
# gamry_kcl_conc, gamry_kcl_sigma, gamry_kcl_errors, _ = \
#     organize_for_conc_plot(benchtop_data, 'KCl', source_filter='Gamry')
#
# gamry_data_kcl = None
# if len(gamry_kcl_sigma) > 0 and len(gamry_kcl_sigma[0]) > 0:
#     gamry_data_kcl = (
#         gamry_kcl_conc,
#         gamry_kcl_sigma[0],  # First temp
#         gamry_kcl_errors[0],
#         'Gamry (293.15 K)'
#     )

fig = plot_study_concentration(
    data=benchtop_data,
    compound='KCl',  # Change to your compound
    output_file=os.path.join(OUTPUT_DIR, 'kcl_vs_concentration.pdf'),
    gamry_data=None,  # Or gamry_data_kcl if available
    show_delta=True,
    show_legend=SHOW_LEGEND,
    show_title=SHOW_TITLE,
    compound_latex=r'KCl',  # LaTeX formatted name
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/kcl_vs_concentration.pdf")
print()

# ========================================
# Example Plot 2: KCl σ vs Temperature
# ========================================

print("Generating Plot 2: KCl Conductivity vs Temperature...")

fig = plot_study_temperature(
    data=benchtop_data,
    compound='KCl',
    output_file=os.path.join(OUTPUT_DIR, 'kcl_vs_temperature.pdf'),
    show_delta=True,
    show_legend=SHOW_LEGEND,
    show_title=SHOW_TITLE,
    compound_latex=r'KCl',
    colormap=COLORMAP_TEMPERATURE,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print(f"  Saved to {OUTPUT_DIR}/kcl_vs_temperature.pdf")
print()

# ========================================
# Example Plot 3: KCl σ vs Pressure (NEW!)
# ========================================

# Uncomment if you have pressure data
# print("Generating Plot 3: KCl Conductivity vs Pressure...")
#
# fig = plot_study_pressure(
#     data=benchtop_data,
#     compound='KCl',
#     output_file=os.path.join(OUTPUT_DIR, 'kcl_vs_pressure.pdf'),
#     show_delta=False,  # No pressure model available yet
#     show_legend=SHOW_LEGEND,
#     show_title=SHOW_TITLE,
#     compound_latex=r'KCl',
#     colormap=COLORMAP_PRESSURE,
#     fontsize_label=FONTSIZE_AXIS_LABEL,
#     fontsize_title=FONTSIZE_TITLE,
#     fontsize_legend=FONTSIZE_LEGEND
# )
#
# print(f"  Saved to {OUTPUT_DIR}/kcl_vs_pressure.pdf")
# print()

# ========================================
# Add More Compounds Here
# ========================================

# Copy and modify the plot blocks above for each compound
# Example: NaBr, MgCl2, etc.

# ========================================
# Summary
# ========================================

print("=" * 70)
print("All plots saved to:", OUTPUT_DIR)
print("=" * 70)
print()

print("Next steps:")
print("  1. Review plots for accuracy")
print("  2. Adjust config_plots.py if needed (fonts, colors, etc.)")
print("  3. Add more compounds by copying the plot blocks above")
print("  4. Add Gamry overlays if impedance data available")
