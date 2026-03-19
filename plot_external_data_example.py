#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for plotting external benchtop probe conductivity data

This demonstrates how to use the new plotting.py module with data from
sources other than Gamry impedance measurements, such as benchtop
conductivity probes.

Adapted from Mahboub et al. (2026, in press) workflow
"""

import numpy as np
from plotting import (
    plot_sigma_vs_concentration,
    plot_sigma_vs_temperature,
    load_external_data
)

# ========================================
# Example 1: Plot NaCl data vs concentration
# ========================================

def example_nacl_vs_concentration():
    """
    Plot NaCl conductivity vs concentration at multiple temperatures.
    Data format matches Mahboub et al. (2026) structure.
    """
    # Concentrations in mol/kg (molal)
    conc_nacl = np.array([10, 30, 50, 75, 100, 150]) / 58.44  # Convert g/kg to mol/kg

    # Temperatures in Celsius
    T_C_nacl = [-10, -6, -3, -1, 5, 20, 25]

    # Conductivity data (mS/cm converted to S/m by multiplying by 0.1)
    # Each row is a replicate, averaged across replicates
    elec_cond_nacl_25 = 0.1 * np.array([
        [17.88, 48.61, 74.90, 105.67, 132.33, 176.55],
        [17.77, 48.75, 75.01, 105.52, 132.77, 176.33],
        [17.53, 48.85, 75.91, 104.95, 132.55, 175.23]
    ])
    elec_cond_nacl_20 = 0.1 * np.array([
        [16.25, 44.19, 68.09, 96.06, 120.3, 160.5],
        [16.15, 44.32, 68.19, 95.93, 120.7, 160.3],
        [15.94, 44.41, 69.01, 95.41, 120.5, 159.3]
    ])
    elec_cond_nacl_5 = 0.1 * np.array([
        [11.88, 31.24, 47.89, 68.13, 85.99, 112.8],
        [12.01, 31.19, 48.23, 69.25, 86.10, 114.9],
        [11.57, 31.60, 49.40, 67.83, 87.87, 112.5]
    ])

    # Calculate means and standard deviations
    def mean_std(a):
        return np.nanmean(a, axis=0), np.nanstd(a, axis=0)

    elec_mean_25, err_25 = mean_std(elec_cond_nacl_25)
    elec_mean_20, err_20 = mean_std(elec_cond_nacl_20)
    elec_mean_5, err_5 = mean_std(elec_cond_nacl_5)

    # Add systematic uncertainties (instrument errors)
    def total_err(std, mean):
        # Combine standard deviation with systematic errors:
        # 0.5% + 2.89% + 0.5% systematic
        return np.sqrt(std**2 + (0.005*mean)**2 + (0.0289*mean)**2 + (0.005*mean)**2)

    err_25 = total_err(err_25, elec_mean_25)
    err_20 = total_err(err_20, elec_mean_20)
    err_5 = total_err(err_5, elec_mean_5)

    # Organize data for plotting
    sigma_data = [elec_mean_5, elec_mean_20, elec_mean_25]
    sigma_errors = [err_5, err_20, err_25]
    temp_labels = ['278.15 K (5°C)', '293.15 K (20°C)', '298.15 K (25°C)']

    # Create plot
    fig = plot_sigma_vs_concentration(
        conc_data=conc_nacl,
        sigma_data=sigma_data,
        sigma_errors=sigma_errors,
        temp_labels=temp_labels,
        xlabel=r'Concentration (mol/kg$_{\mathrm{H_2O}}$)',
        title='NaCl Conductivity vs Concentration',
        show_delta=False,
        out_file='NaCl_sigma_vs_conc.pdf',
        colormap='viridis'
    )

    print("NaCl vs concentration plot saved to: NaCl_sigma_vs_conc.pdf")


# ========================================
# Example 2: Plot MgSO4 data vs temperature
# ========================================

def example_mgso4_vs_temperature():
    """
    Plot MgSO4 conductivity vs temperature at multiple concentrations.
    """
    # Concentrations in mol/kg (molal)
    conc_mgso4 = np.array([3, 40, 80, 120, 170, 200]) / 120.366

    # Temperatures in Kelvin
    T_K_mgso4 = np.array([-10, -6, -3, -1, 5, 20, 25]) + 273.15

    # Conductivity data (averaged across replicates)
    # Each column is one concentration, rows are temperatures
    # This is transposed data organized by concentration
    sigma_by_temp = 0.1 * np.array([
        # -10°C data
        [0.000, 11.09, 15.50, 20.37, 21.88, 22.31],
        # -6°C
        [1.843, 12.05, 18.12, 22.75, 24.42, 24.01],
        # -3°C
        [1.683, 12.78, 19.35, 24.71, 27.71, 27.77],
        # -1°C
        [1.930, 14.12, 21.14, 26.32, 29.16, 29.48],
        # 5°C
        [2.209, 15.99, 25.19, 31.11, 35.00, 35.43],
        # 20°C
        [3.112, 22.94, 36.42, 44.40, 51.34, 53.05],
        # 25°C
        [3.423, 25.24, 40.06, 48.85, 56.47, 58.35]
    ])

    # Transpose to get conductivity arrays for each concentration
    sigma_by_conc = [sigma_by_temp[:, i] for i in range(len(conc_mgso4))]

    # Create labels
    conc_labels = [f'{c:.3f} mol/kg' for c in conc_mgso4]

    # Create plot
    fig = plot_sigma_vs_temperature(
        temp_data=T_K_mgso4,
        sigma_data=sigma_by_conc,
        conc_labels=conc_labels,
        sigma_errors=None,  # Can add errors if available
        xlabel=r'Temperature (K)',
        title=r'MgSO$_4$ Conductivity vs Temperature',
        show_delta=False,
        out_file='MgSO4_sigma_vs_temp.pdf',
        colormap='plasma'
    )

    print("MgSO4 vs temperature plot saved to: MgSO4_sigma_vs_temp.pdf")


# ========================================
# Example 3: Load data from CSV file
# ========================================

def example_load_from_csv():
    """
    Demonstrate loading external data from CSV file.

    Expected CSV format:
    concentration,temperature,conductivity,error
    0.1,273.15,1.5,0.05
    0.2,273.15,2.8,0.08
    ...
    """
    # This is just an example - file would need to exist
    # Uncomment to use:
    #
    # data = load_external_data('my_probe_data.csv', format='csv')
    #
    # # Group by temperature
    # unique_temps = np.unique(data['temperature'])
    # sigma_data = []
    # sigma_errors = []
    #
    # for temp in unique_temps:
    #     mask = data['temperature'] == temp
    #     sigma_data.append(data['conductivity'][mask])
    #     sigma_errors.append(data['errors'][mask])
    #
    # fig = plot_sigma_vs_concentration(
    #     conc_data=data['concentration'][data['temperature'] == unique_temps[0]],
    #     sigma_data=sigma_data,
    #     sigma_errors=sigma_errors,
    #     temp_labels=[f'{t:.1f} K' for t in unique_temps],
    #     out_file='my_data_vs_conc.pdf'
    # )

    print("CSV loading example (commented out - create CSV file to use)")


# ========================================
# Example 4: Integration with Gamry data
# ========================================

def example_gamry_integration():
    """
    Show how to plot Gamry measurements alongside external data.
    """
    # Example: You have Gamry measurements loaded
    # from gamryTools import Solution
    #
    # gamry_measurements = [...]  # List of Solution objects
    #
    # # Extract Gamry data
    # gamry_concs = [m.w_molal for m in gamry_measurements if hasattr(m, 'w_molal')]
    # gamry_sigmas = [m.sigma_Sm for m in gamry_measurements if hasattr(m, 'sigma_Sm')]
    #
    # # External benchtop data
    # external_concs = np.array([0.5, 1.0, 1.5])
    # external_sigmas = np.array([4.5, 8.2, 11.5])
    #
    # # Plot both on same figure
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(external_concs, external_sigmas, 'o-', label='Benchtop probe')
    # ax.plot(gamry_concs, gamry_sigmas, 's-', label='Gamry impedance')
    # ax.set_xlabel('Concentration (mol/kg)')
    # ax.set_ylabel('Conductivity (S/m)')
    # ax.legend()
    # fig.savefig('gamry_vs_benchtop.pdf')

    print("Gamry integration example (requires loaded Gamry data)")


# ========================================
# Main execution
# ========================================

if __name__ == '__main__':
    print("=" * 60)
    print("HiPOZ External Data Plotting Examples")
    print("=" * 60)
    print()

    print("Running Example 1: NaCl vs Concentration...")
    example_nacl_vs_concentration()
    print()

    print("Running Example 2: MgSO4 vs Temperature...")
    example_mgso4_vs_temperature()
    print()

    print("Example 3: CSV Loading (see code for template)")
    example_load_from_csv()
    print()

    print("Example 4: Gamry Integration (see code for template)")
    example_gamry_integration()
    print()

    print("=" * 60)
    print("Examples complete! Check generated PDF files.")
    print("=" * 60)
