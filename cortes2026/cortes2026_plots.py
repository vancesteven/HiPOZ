#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication plots for Cortes et al. (2026).

This script generates conductivity vs concentration and conductivity vs temperature
plots for benchtop probe measurements integrated with Gamry impedance data.

Data sources:
1. Benchtop probe data: Cortes2026BenchtopData.csv (TO BE CREATED)
2. Gamry impedance data: ../JesusCortes/Data/ (multiple dates)
3. McCleskey et al. (2011) reference data

Outputs:
- Plots in cortes_plots/ directory
- Automatic overlay of Gamry high-pressure measurements on benchtop data
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Generalized study plotting functions
from study_plots import (
    load_study_data,
    plot_study_concentration,
    plot_study_temperature
)

# Gamry impedance data loading (using cortes_data_processing)
import cortes_data_processing as cdp

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
# Helpers
# ========================================

def safe_filename_stem(compound):
    """
    Build a filesystem-safe filename stem from a compound name.

    Replaces characters that are invalid on Windows (notably ':' in ratio
    names like 'Na2SO4:KCl_2:1') with '-', and spaces with '_'.
    """
    return (compound.lower()
            .replace(' ', '_')
            .replace(':', '-'))


# ========================================
# Configuration
# ========================================

# Control EIS (Gamry) data overlay
# Set to True to include 1 bar EIS measurements on benchtop plots
# Set to False to show only benchtop data
INCLUDE_EIS_OVERLAY = True

# Pressure filter for EIS overlay (MPa)
# 1 bar ≈ 0.1 MPa, use 1.0 MPa to capture atmospheric pressure measurements
EIS_PRESSURE_MAX = 1.0  # MPa

# Get paths relative to parent directory (hipozgenai/)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Gamry data directories from JesusCortes/Data folder
GAMRY_DATA_DIRS = [
    os.path.join(PARENT_DIR, 'data', '20250813Cortes'),
    os.path.join(PARENT_DIR, 'data', '20250814Cortes'),
    os.path.join(PARENT_DIR, 'data', '20250815Cortes'),
    os.path.join(PARENT_DIR, 'data', 'RoseData'),  # Additional NaCl measurements
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'cortes_plots')  # Relative to this script

# Note: Gamry data is currently in ../JesusCortes/Data/{08.12.25, 08.13.25, 08.14.25, 08.15.25}
# Will need to reorganize into ../data/ folders following HiPOZ convention

# Analysis workflow:
# 1. Clean benchtop data from JesusData2025.csv → Cortes2026BenchtopData.csv
# 2. Organize Gamry data into data/ folders
# 3. Run HiPOZ impedance analysis: python gamry_HiPOZ.py --dates 20250813Cortes 20250815Cortes
# 4. Results saved to: data/*/hipoz_*_results.csv
# 5. This script loads and combines results from all datasets

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# Helper Functions
# ========================================

def get_mixture_ion_spec(compound):
    """
    Get ion specification for mixture compounds to enable McCleskey predictions.

    For mixtures like Na2SO4:KCl_2:1, parse the ratio and create combined ion spec.
    Assumes ratio is by molar concentration.

    Returns
    -------
    ion_spec : dict or None
        Ion specification for McCleskey model, or None if not a supported mixture
    """
    # Na2SO4:KCl mixtures
    if compound.startswith('Na2SO4:KCl_'):
        ratio_str = compound.split('_')[-1]  # e.g., "2:1"
        try:
            ratio_parts = [float(x) for x in ratio_str.split(':')]
            na2so4_frac = ratio_parts[0] / sum(ratio_parts)
            kcl_frac = ratio_parts[1] / sum(ratio_parts)

            # Combined ion spec (relative to 1 mol/kg total)
            ion_spec = {
                'Na_p1': 2.0 * na2so4_frac,  # 2 Na+ per Na2SO4
                'SO4_m2': na2so4_frac,       # 1 SO4²⁻ per Na2SO4
                'K_p1': kcl_frac,            # 1 K+ per KCl
                'Cl_m1': kcl_frac            # 1 Cl⁻ per KCl
            }
            return ion_spec
        except:
            return None

    # NaCl:MgSO4 mixtures
    if compound.startswith('NaCl:MgSO4_'):
        ratio_str = compound.split('_')[-1]
        try:
            ratio_parts = [float(x) for x in ratio_str.split(':')]
            nacl_frac = ratio_parts[0] / sum(ratio_parts)
            mgso4_frac = ratio_parts[1] / sum(ratio_parts)

            ion_spec = {
                'Na_p1': nacl_frac,          # 1 Na+ per NaCl
                'Cl_m1': nacl_frac,          # 1 Cl⁻ per NaCl
                'Mg_p2': mgso4_frac,         # 1 Mg²+ per MgSO4
                'SO4_m2': mgso4_frac         # 1 SO4²⁻ per MgSO4
            }
            return ion_spec
        except:
            return None

    return None

def extract_compound_overlay_data(gamry_df, compound, pressure_max=None):
    """
    Extract Gamry overlay data for a specific compound.

    Parameters
    ----------
    gamry_df : pandas.DataFrame
        Gamry measurement data
    compound : str
        Compound name
    pressure_max : float, optional
        Maximum pressure (MPa) for filtering. If None, include all pressures.

    Returns
    -------
    list or None
        Format expected by plot_study_concentration:
        [conc_array, sigma_array, sigma_err_array, label_string]
        or None if no data available
    """
    if gamry_df is None:
        return None

    # Map compound names (handle different naming conventions)
    compound_map = {
        'NaCl': 'NaCl',
        'KCl': 'KCl',
        'MgSO4': 'MgSO4',
        'Na2SO4': 'Na2SO4',
        'NaCl:MgSO4_1:1': 'NaCl:MgSO4',
        'NaCl:MgSO4_2:1': 'NaCl:MgSO4',
        'NaCl:MgSO4_1:2': 'NaCl:MgSO4',
    }

    # Try direct match first
    comp_data = gamry_df[gamry_df['comp'] == compound]

    # Try mapped name if no direct match
    if len(comp_data) == 0 and compound in compound_map:
        comp_data = gamry_df[gamry_df['comp'] == compound_map[compound]]

    if len(comp_data) == 0:
        return None

    # Filter by pressure if requested
    if pressure_max is not None and 'P_MPa' in comp_data.columns:
        comp_data = comp_data[comp_data['P_MPa'] <= pressure_max]
        if len(comp_data) == 0:
            return None

    # Extract arrays and convert to numeric types
    # Note: w_molal might be stored as strings in CSV, so convert explicitly
    conc_molal = pd.to_numeric(comp_data['w_molal'], errors='coerce').values
    sigma_Sm = comp_data['conductivity_Sm'].values

    # Remove NaN concentrations
    valid_mask = ~np.isnan(conc_molal)
    conc_molal = conc_molal[valid_mask]
    sigma_Sm = sigma_Sm[valid_mask]

    if len(conc_molal) == 0:
        return None

    # Get error bars (use SEM if available, otherwise 5%)
    if 'conductivity_sem' in comp_data.columns:
        sigma_err = comp_data['conductivity_sem'].values[valid_mask]
        # Fill NaN with 5% uncertainty
        sigma_err = np.where(np.isnan(sigma_err), 0.05 * sigma_Sm, sigma_err)
    else:
        sigma_err = 0.05 * sigma_Sm

    # Generate label
    if pressure_max is not None:
        label = f'EIS (P ≤ {pressure_max:.1f} MPa)'
    else:
        label = 'EIS'

    return [conc_molal, sigma_Sm, sigma_err, label]

# ========================================
# Load Data
# ========================================

print("=" * 70)
print("Cortes et al. (2026) - Conductivity Plots")
print("=" * 70)
print()

# Load benchtop data from CSV
benchtop_file = os.path.join(os.path.dirname(__file__), 'Cortes2026BenchtopData.csv')
if not os.path.exists(benchtop_file):
    print(f"WARNING: {benchtop_file} not found!")
    print("  Need to create this file from JesusData2025.csv")
    print("  Run: cd cortes2026 && python parse_benchtop_data.py")
    print()
    print("Exiting - please create benchtop data file first.")
    sys.exit(1)

print(f"Loading benchtop data from {benchtop_file}...")
benchtop_data = load_study_data(benchtop_file)
print(f"  Found {len(benchtop_data)} compounds")
print()

# Load Gamry impedance analysis results from zAnalysis CSV files
print("Loading Gamry impedance data from zAnalysis files...")
# Set working directory to parent for data loading
import os as _os
original_dir = _os.getcwd()
_os.chdir(PARENT_DIR)
gamry_data_dirs = ['20250813Cortes', '20250814Cortes', '20250815Cortes']
gamry_df_raw = cdp.load_cortes_data(gamry_data_dirs)
_os.chdir(original_dir)

if gamry_df_raw is not None:
    # Average replicates
    gamry_df = cdp.average_replicates(gamry_df_raw)
    print(f"  Combined Gamry data: {len(gamry_df)} measurements")
    print(f"  Compounds: {', '.join(sorted(gamry_df['comp'].unique()))}")
    print()
else:
    gamry_df = None
    print("  No Gamry impedance data found (benchtop-only plots)")
    print()

# ========================================
# Generate Plots
# ========================================

# Compounds to analyze (update based on actual benchtop data)
# Expected: KCl, NaCl, MgSO4, Na2SO4, mixtures, organic acids
compounds_to_plot = []

# Detect available compounds from benchtop data
# Exclude certain amino acids from plotting (but keep in tables)
exclude_from_plots = ['Alanine', 'Glutamic Acid', 'Aspartic Acid']

if benchtop_data:
    all_compounds = list(benchtop_data.keys())
    compounds_to_plot = [c for c in all_compounds if c not in exclude_from_plots]
    print(f"Detected compounds: {', '.join(all_compounds)}")
    print(f"Plotting: {', '.join(compounds_to_plot)}")
    print(f"Excluded from plots: {', '.join([c for c in all_compounds if c in exclude_from_plots])}")
    print()

# Plot each compound
plot_number = 1
for compound in compounds_to_plot:
    # Determine LaTeX formatting for compound name
    compound_latex = compound
    if compound == 'KCl':
        compound_latex = 'KCl'
    elif compound == 'NaCl':
        compound_latex = 'NaCl'
    elif compound == 'MgSO4':
        compound_latex = r'MgSO$_4$'
    elif compound == 'Na2SO4':
        compound_latex = r'Na$_2$SO$_4$'
    elif compound == 'Na2CO3':
        compound_latex = r'Na$_2$CO$_3$'
    elif 'NaCl+MgSO4' in compound or 'NaCl:MgSO4' in compound:
        compound_latex = r'NaCl:MgSO$_4$'
    elif compound.startswith('Na2SO4:KCl_'):
        # Format Na2SO4:KCl mixtures with proper subscripts and replace _ with space
        ratio = compound.split('_')[-1]  # Extract ratio like "2:1"
        compound_latex = rf'Na$_2$SO$_4$:KCl {ratio}'
    elif compound == 'NaCl+Glycine':
        compound_latex = 'NaCl+Glycine'

    # Determine if we should show McCleskey model (Δ%)
    # Show for single salts and salt mixtures, but not for organic compounds
    show_delta_plot = False
    mixture_ion_spec = None

    # Single salts - use default ion specs
    if compound in ['KCl', 'NaCl', 'MgSO4', 'Na2SO4', 'NH4Cl', 'Na2CO3']:
        show_delta_plot = True

    # Salt mixtures - use custom ion specs (but not glycine mixtures)
    elif compound.startswith(('Na2SO4:KCl_', 'NaCl:MgSO4_')):
        mixture_ion_spec = get_mixture_ion_spec(compound)
        show_delta_plot = (mixture_ion_spec is not None)

    # Organic compounds (amino acids, glycine mixtures) - no McCleskey
    # (show_delta_plot stays False)

    # Plot 1: σ vs Concentration
    print(f"Generating Plot {plot_number}: {compound} Conductivity vs Concentration...")

    # Extract EIS overlay data if enabled
    gamry_data_comp = None
    if INCLUDE_EIS_OVERLAY and gamry_df is not None:
        gamry_data_comp = extract_compound_overlay_data(
            gamry_df, compound, pressure_max=EIS_PRESSURE_MAX
        )
        if gamry_data_comp:
            print(f"  EIS overlay: {len(gamry_data_comp[0])} points at P ≤ {EIS_PRESSURE_MAX} MPa")

    plot_study_concentration(
        data=benchtop_data,
        compound=compound,
        output_file=os.path.join(OUTPUT_DIR, f'{safe_filename_stem(compound)}_vs_concentration.pdf'),
        gamry_data=gamry_data_comp,
        show_delta=show_delta_plot,
        compound_latex=compound_latex,
        colormap=COLORMAP_CONCENTRATION,
        fontsize_label=FONTSIZE_AXIS_LABEL,
        fontsize_title=FONTSIZE_TITLE,
        fontsize_legend=FONTSIZE_LEGEND,
        ion_spec=mixture_ion_spec
    )
    print(f"  Saved to {OUTPUT_DIR}/{safe_filename_stem(compound)}_vs_concentration.pdf")
    if gamry_data_comp:
        print(f"  Overlay: {len(gamry_data_comp[0])} Gamry points added")
    print()
    plot_number += 1

    # Plot 2: σ vs Temperature
    print(f"Generating Plot {plot_number}: {compound} Conductivity vs Temperature...")
    plot_study_temperature(
        data=benchtop_data,
        compound=compound,
        output_file=os.path.join(OUTPUT_DIR, f'{safe_filename_stem(compound)}_vs_temperature.pdf'),
        show_delta=show_delta_plot,
        compound_latex=compound_latex,
        colormap=COLORMAP_TEMPERATURE,
        fontsize_label=FONTSIZE_AXIS_LABEL,
        fontsize_title=FONTSIZE_TITLE,
        fontsize_legend=FONTSIZE_LEGEND,
        ion_spec=mixture_ion_spec
    )
    print(f"  Saved to {OUTPUT_DIR}/{safe_filename_stem(compound)}_vs_temperature.pdf")
    print()
    plot_number += 1

# ========================================
# Summary Statistics
# ========================================

print("=" * 70)
print("Summary Statistics")
print("=" * 70)
print()

import numpy as np

for compound in compounds_to_plot:
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
print(f"  - Benchtop probe data: {benchtop_file}")
print("  - Gamry impedance overlay: JesusCortes/Data/")
print(f"  - All figures: {OUTPUT_DIR}/*_vs_*.pdf")
print()
print("Citation:")
print("  Cortes, J., et al. (2026)")
print()
print("Key features:")
print("  - Data stored in easy-to-edit CSV format")
print("  - Automatic integration with Gamry high-pressure measurements")
print("  - Includes organic compound measurements (amino acids)")
print("  - Pressure range up to 600 MPa")
print()
