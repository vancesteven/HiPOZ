#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication plots for Mahboub et al. (2026, in press).

This script generates conductivity vs concentration and conductivity vs temperature
plots for benchtop probe measurements integrated with Gamry impedance data.

Data sources:
1. Benchtop probe data: Mahboub2026BenchtopData.csv
2. Gamry impedance data: data/20250813Mahboub2026 and data/20250815Mahboub2026
3. McCleskey et al. (2011) reference data

Outputs:
- 10 plots (5 compounds × 2 plot types) in mahboub_plots/ directory
- Automatic overlay of Gamry measurements on benchtop data
"""

import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Generalized study plotting functions
from study_plots import (
    load_study_data,
    plot_study_concentration,
    plot_study_temperature
)

# Gamry impedance integration
from gamry_integration import (
    load_gamry_results,
    extract_compound_overlay
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

# McCleskey (2012) applicability limits (mol/kg_H2O)
# These mark the concentration range where the McCleskey model is validated
MCCLESKEY_LIMITS = {
    'NaCl': 0.9999,
    'MgSO4': 0.01245,
    'NH4Cl': 1.034,
    'Na2CO3': 0.3041
}

# Get paths relative to parent directory (hipozgenai/)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMRY_DATA_DIRS = [
    os.path.join(PARENT_DIR, 'data', '20250813Mahboub2026'),
    os.path.join(PARENT_DIR, 'data', '20250815Mahboub2026')
]
OUTPUT_DIR = 'mahboub_plots'  # Relative to this script (in mahboub2026/)

# Analysis workflow:
# 1. Run HiPOZ impedance analysis for both datasets:
#    python gamry_HiPOZ.py --dates 20250813Mahboub2026 20250815Mahboub2026
# 2. Results saved to: data/*/hipoz_*_results.csv
# 3. This script loads and combines results from both datasets
# 4. Benchtop data loaded separately from Mahboub2026BenchtopData.csv

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

# Load Gamry impedance analysis results from multiple datasets
gamry_dfs = []
for data_dir in GAMRY_DATA_DIRS:
    df = load_gamry_results(data_dir, verbose=True)
    if df is not None:
        gamry_dfs.append(df)

# Combine all Gamry data
if gamry_dfs:
    gamry_df = pd.concat(gamry_dfs, ignore_index=True)
    print(f"Combined Gamry data: {len(gamry_df)} total measurements from {len(gamry_dfs)} dataset(s)")
    print()
else:
    gamry_df = None

# ========================================
# Generate Plots
# ========================================

# Plot 1: NaCl σ vs Concentration
print("Generating Plot 1: NaCl Conductivity vs Concentration...")
gamry_data_nacl = extract_compound_overlay(gamry_df, 'NaCl')
plot_study_concentration(
    data=benchtop_data,
    compound='NaCl',
    output_file=os.path.join(OUTPUT_DIR, 'nacl_vs_concentration.pdf'),
    gamry_data=gamry_data_nacl,
    show_delta=True,
    mccleskey_limit=MCCLESKEY_LIMITS.get('NaCl'),
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)
print(f"  Saved to {OUTPUT_DIR}/nacl_vs_concentration.pdf")
if gamry_data_nacl:
    print(f"  Overlay: {len(gamry_data_nacl[0])} Gamry points added")
print()

# Plot 2: NaCl σ vs Temperature
print("Generating Plot 2: NaCl Conductivity vs Temperature...")
plot_study_temperature(
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

# Plot 3: MgSO4 σ vs Concentration
print("Generating Plot 3: MgSO4 Conductivity vs Concentration...")
gamry_data_mgso4 = extract_compound_overlay(gamry_df, 'MgSO4')
plot_study_concentration(
    data=benchtop_data,
    compound='MgSO4',
    output_file=os.path.join(OUTPUT_DIR, 'mgso4_vs_concentration.pdf'),
    gamry_data=gamry_data_mgso4,
    show_delta=True,
    compound_latex=r'MgSO$_4$',
    mccleskey_limit=MCCLESKEY_LIMITS.get('MgSO4'),
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)
print(f"  Saved to {OUTPUT_DIR}/mgso4_vs_concentration.pdf")
if gamry_data_mgso4:
    print(f"  Overlay: {len(gamry_data_mgso4[0])} Gamry points added")
print()

# Plot 4: MgSO4 σ vs Temperature
print("Generating Plot 4: MgSO4 Conductivity vs Temperature...")
plot_study_temperature(
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

# Plot 5: NH4Cl σ vs Concentration
print("Generating Plot 5: NH4Cl Conductivity vs Concentration...")
plot_study_concentration(
    data=benchtop_data,
    compound='NH4Cl',
    output_file=os.path.join(OUTPUT_DIR, 'nh4cl_vs_concentration.pdf'),
    show_delta=True,
    compound_latex=r'NH$_4$Cl',
    mccleskey_limit=MCCLESKEY_LIMITS.get('NH4Cl'),
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)
print(f"  Saved to {OUTPUT_DIR}/nh4cl_vs_concentration.pdf")
print()

# Plot 6: NH4Cl σ vs Temperature
print("Generating Plot 6: NH4Cl Conductivity vs Temperature...")
plot_study_temperature(
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

# Plot 7: Na2CO3 σ vs Concentration
print("Generating Plot 7: Na2CO3 Conductivity vs Concentration...")
plot_study_concentration(
    data=benchtop_data,
    compound='Na2CO3',
    output_file=os.path.join(OUTPUT_DIR, 'na2co3_vs_concentration.pdf'),
    show_delta=True,
    compound_latex=r'Na$_2$CO$_3$',
    mccleskey_limit=MCCLESKEY_LIMITS.get('Na2CO3'),
    colormap=COLORMAP_CONCENTRATION,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)
print(f"  Saved to {OUTPUT_DIR}/na2co3_vs_concentration.pdf")
print()

# Plot 8: Na2CO3 σ vs Temperature
print("Generating Plot 8: Na2CO3 Conductivity vs Temperature...")
plot_study_temperature(
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

# Plot 9: Mixture σ vs Concentration
print("Generating Plot 9: Mixture (1:1:1) Conductivity vs Concentration...")
plot_study_concentration(
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

# Plot 10: Mixture σ vs Temperature
print("Generating Plot 10: Mixture (1:1:1) Conductivity vs Temperature...")
plot_study_temperature(
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

import numpy as np

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
    if 'comp' in gamry_df.columns:
        for comp in gamry_df['comp'].dropna().unique():
            n_comp = len(gamry_df[gamry_df['comp'] == comp])
            print(f"  {comp}: {n_comp} measurements")
    print()

print("=" * 70)
print("All plots saved to:", OUTPUT_DIR)
print("=" * 70)
print()

print("Publication-ready plots:")
print("  - Benchtop probe data: Mahboub2026BenchtopData.csv")
print("  - Gamry impedance overlay: data/20250815Mahboub2026")
print("  - All figures: mahboub_plots/*_vs_*.pdf")
print()
print("Citation:")
print("  Mahboub, R., et al. (2026)")
print()
print("Key features:")
print("  - Data stored in easy-to-edit CSV format")
print("  - Automatic integration with Gamry impedance measurements")
print("  - Reusable plotting functions (gamry_integration.py)")
print("  - Command-line scriptable for reproducibility")
