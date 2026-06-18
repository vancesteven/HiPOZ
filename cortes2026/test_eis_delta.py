#!/usr/bin/env python3
"""
Test script to verify EIS data appears in delta subplot.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from study_plots import load_study_data, plot_study_concentration, compute_mccleskey_model
import cortes_data_processing as cdp
import numpy as np
import pandas as pd

# Load benchtop data
benchtop_file = 'Cortes2026BenchtopData.csv'
benchtop_data = load_study_data(benchtop_file)

# Load Gamry data
os.chdir('..')
gamry_df_raw = cdp.load_cortes_data(['20250813Cortes', '20250814Cortes', '20250815Cortes'])
gamry_df = cdp.average_replicates(gamry_df_raw)
os.chdir('cortes2026')

# Filter for NaCl at low pressure
nacl_eis = gamry_df[gamry_df['comp'] == 'NaCl']
nacl_1bar = nacl_eis[nacl_eis['P_MPa'] <= 1.0]

# Convert to numeric and remove NaN
conc = pd.to_numeric(nacl_1bar['w_molal'], errors='coerce').values
sigma = nacl_1bar['conductivity_Sm'].values
valid = ~np.isnan(conc)
conc = conc[valid]
sigma = sigma[valid]

print("EIS Data for NaCl (P ≤ 1 MPa):")
print(f"Concentrations: {conc}")
print(f"Conductivity: {sigma}")
print()

# Compute McCleskey model
model = compute_mccleskey_model(conc, [293.15], compound='NaCl')[0]
print(f"McCleskey model: {model}")
print()

# Compute delta
delta = 100.0 * (sigma - model) / model
print(f"Δ% from McCleskey: {delta}")
print()

# Generate test plot
if 'conductivity_sem' in nacl_1bar.columns:
    err = nacl_1bar['conductivity_sem'].values[valid]
else:
    err = 0.05 * sigma

gamry_data = [conc, sigma, err, 'EIS (P ≤ 1.0 MPa)']

plot_study_concentration(
    data=benchtop_data,
    compound='NaCl',
    output_file='test_nacl_eis_delta.pdf',
    gamry_data=gamry_data,
    show_delta=True,
    compound_latex='NaCl',
    show_title=False
)

print("\nTest plot saved to: test_nacl_eis_delta.pdf")
print("Check if EIS points (black circles) appear in both:")
print("  1. Main plot (top panel)")
print("  2. Delta plot (bottom panel)")
