#!/usr/bin/env python3
"""
Generate the correct Mahboub2026BenchtopData.csv from the raw data in MahboubEtAl2026.py
"""
import numpy as np
import csv

# NaCl data
conc_nacl = np.array([10, 30, 50, 75, 100, 150]) / 58.44
T_C_nacl = [-10, -6, -3, -1, 5, 20, 25]

elec_cond_nacl_25 = 0.1*np.array([
    [17.88, 48.61, 74.90, 105.67, 132.33, 176.55],
    [17.77, 48.75, 75.01, 105.52, 132.77, 176.33],
    [17.53, 48.85, 75.91, 104.95, 132.55, 175.23]
])
elec_cond_nacl_20 = 0.1*np.array([
    [16.25, 44.19, 68.09, 96.06, 120.3, 160.5],
    [16.15, 44.32, 68.19, 95.93, 120.7, 160.3],
    [15.94, 44.41, 69.01, 95.41, 120.5, 159.3]
])
elec_cond_nacl_5 = 0.1*np.array([
    [11.88, 31.24, 47.89, 68.13, 85.99, 112.8],
    [12.01, 31.19, 48.23, 69.25, 86.10, 114.9],
    [11.57, 31.60, 49.40, 67.83, 87.87, 112.5]
])
elec_cond_nacl_minus1 = 0.1*np.array([
    [9.843, 26.97, 40.33, 57.44, 71.01, 94.74],
    [9.862, 26.49, 42.00, 56.78, 71.08, 95.81],
    [9.835, 27.51, 41.08, 56.64, 73.71, 95.80]
])
elec_cond_nacl_minus3 = 0.1*np.array([
    [10.33, 25.48, 38.18, 54.68, 70.74, 93.73],
    [9.593, 25.32, 40.45, 54.73, 69.13, 92.16],
    [9.547, 24.35, 39.69, 54.97, 70.31, 91.05]
])
elec_cond_nacl_minus6 = 0.1*np.array([
    [8.755, 23.62, 35.53, 50.67, 62.60, 84.79],
    [8.6485,23.89, 35.04, 50.89, 64.00, 85.79],
    [8.542, 23.09, 35.89, 51.86, 64.85, 85.29]
])
elec_cond_nacl_minus10 = 0.1*np.array([
    [7.852, 22.41, 32.26, 43.24, 57.06, 75.76],
    [7.852, 21.95, 31.87, 45.55, 56.45, 79.25],
    [7.852, 20.87, 31.33, 44.73, 58.90, 74.91]
])

# MgSO4 data
conc_mgso4 = np.array([3, 40, 80, 120, 170, 200]) / 120.366
T_C_mgso4 = [-10, -6, -3, -1, 5, 20, 25]

elec_cond_mgso4_25 = 0.1*np.array([
    [3.318, 25.01, 40.03, 49.91, 56.52, 57.99],
    [3.443, 25.26, 40.04, 50.15, 56.33, 59.28],
    [3.508, 25.44, 40.11, 46.48, 56.56, 57.79]
])
elec_cond_mgso4_20 = 0.1*np.array([
    [3.016, 22.74, 36.39, 45.37, 51.38, 52.72],
    [3.130, 22.96, 36.40, 45.59, 51.21, 53.89],
    [3.189, 23.13, 36.46, 42.25, 51.42, 52.54]
])
elec_cond_mgso4_5 = 0.1*np.array([
    [2.245, 16.16, 25.05, 31.01, 35.43, 35.71],
    [2.126, 15.89, 25.31, 31.40, 34.77, 35.29],
    [2.255, 15.93, 25.20, 30.92, 34.80, 35.29]
])
elec_cond_mgso4_minus1 = 0.1*np.array([
    [1.997, 14.26, 21.00, 26.26, 28.97, 29.86],
    [1.836, 13.55, 21.01, 26.53, 29.32, 29.89],
    [1.958, 13.55, 21.42, 26.17, 29.20, 28.70]
])
elec_cond_mgso4_minus3 = 0.1*np.array([
    [1.190, 13.10, 19.80, 24.52, 27.80, 27.75],
    [1.753, 12.57, 18.60, 24.73, 26.63, 27.70],
    [2.106, 12.72, 19.62, 24.89, 28.69, 27.85]
])
elec_cond_mgso4_minus6 = 0.1*np.array([
    [1.843, 12.05, 18.12, 22.75, 24.42, 24.01],
    [1.737, 11.65, 18.47, 22.20, 23.87, 25.08],
    [2.086, 11.47, 18.04, 21.84, 24.88, 25.05]
])
elec_cond_mgso4_minus10 = 0.1*np.array([
    [0.000, 11.09, 15.50, 20.37, 21.88, 22.31],
    [0.000, 11.09, 15.50, 20.37, 21.87, 21.98],
    [0.000, 11.09, 15.50, 20.37, 21.61, 22.77]
])

# NH4Cl data
conc_nh4cl = np.array([10, 50, 75, 100]) / 53.491
T_C_nh4cl = [-10, -6, -3, -1, 5, 20, 25]

elec_cond_nh4cl_25 = np.array([
    [2.3331, 10.3466, 14.938, 19.261],
    [2.3177, 10.3642, 14.894, 19.030],
    [2.3320, 10.4170, 14.718, 19.085]
])
elec_cond_nh4cl_20 = 0.1*np.array([
    [21.21, 94.06, 135.8, 175.1],
    [21.07, 94.22, 135.4, 173.0],
    [21.20, 94.70, 133.8, 173.5]
])
elec_cond_nh4cl_5 = 0.1*np.array([
    [15.35, 68.43, 100.0, 131.3],
    [15.37, 69.79, 99.93, 130.9],
    [15.76, 69.04, 99.21, 129.4]
])
elec_cond_nh4cl_minus1 = 0.1*np.array([
    [13.55, 58.97, 86.56, 115.4],
    [13.12, 60.68, 85.68, 113.3],
    [13.36, 59.83, 85.84, 109.2]
])
elec_cond_nh4cl_minus3 = 0.1*np.array([
    [12.53, 54.91, 81.88, 107.9],
    [12.58, 56.64, 82.93, 104.6],
    [12.70, 56.02, 82.03, 106.7]
])
elec_cond_nh4cl_minus6 = 0.1*np.array([
    [12.24, 52.07, 77.13, 100.4],
    [11.40, 54.28, 76.53, 98.49],
    [11.99, 51.78, 75.50, 98.77]
])
elec_cond_nh4cl_minus10 = 0.1*np.array([
    [10.68, 47.00, 69.78, 88.52],
    [8.184, 48.60, 70.18, 89.49],
    [10.36, 46.95, 68.30, 93.01]
])

# Na2CO3 data
conc_na2co3 = np.array([5, 10, 25, 40, 55]) / 105.9888
T_C_na2co3_all = [-10, -6, -3, -1, 5, 20, 25]

elec_cond_na2co3_25 = 0.1*np.array([
    [8.1532, 14.718, 30.547, 44.517, 54.681],
    [8.1972, 14.685, 31.009, 43.703, 54.736],
    [8.3160, 14.564, 30.712, 43.439, 54.120]
])
elec_cond_na2co3_20 = 0.1*np.array([
    [7.412, 13.38, 27.77, 40.47, 49.71],
    [7.452, 13.35, 28.19, 39.73, 49.76],
    [7.560, 13.24, 27.92, 39.49, 49.20]
])
elec_cond_na2co3_5 = 0.1*np.array([
    [5.210, 9.306, 18.94, 26.91, 33.79],
    [5.123, 9.099, 19.00, 26.70, 33.97],
    [5.095, 9.031, 18.98, 26.70, 33.84]
])
elec_cond_na2co3_minus1 = 0.1*np.array([
    [6.495, 11.720, 24.336, 35.440, 43.561],
    [6.531, 11.690, 24.722, 34.785, 43.609],
    [6.624, 11.592, 24.460, 34.570, 43.043]
])
elec_cond_na2co3_minus3 = 0.1*np.array([
    [6.142, 11.088, 23.030, 33.570, 41.264],
    [6.176, 11.059, 23.392, 32.947, 41.310],
    [6.262, 10.966, 23.138, 32.741, 40.771]
])
elec_cond_na2co3_minus6 = 0.1*np.array([
    [3.971, 6.548, 13.99, 19.59, 24.34],
    [3.785, 6.521, 14.31, 19.81, 24.11],
    [3.602, 6.574, 13.87, 19.30, 24.09]
])
elec_cond_na2co3_minus10 = 0.1*np.array([
    [2.9648, 5.3520, 11.108, 16.188, 19.884],
    [2.9808, 5.3400, 11.276, 15.892, 19.904],
    [3.0240, 5.2960, 11.168, 15.796, 19.680]
])

# Mixture data (1:1:1 MgSO4:NaCl:Na2CO3)
conc_mixt = np.array([0.05, 0.1, 0.4, 0.7])
T_C_mixt = [-10, -1, 20, 25]

elec_cond_mixt_25 = (np.array([
    [15.158, 28.017, 72.688, 62.458],
    [15.015, 26.730, 73.711, 66.418],
    [15.191, 26.389, 74.129, 65.230]
]) / 10.0)
elec_cond_mixt_20 = (np.array([
    [13.78, 25.47, 66.08, 56.78],
    [13.65, 24.30, 67.01, 60.38],
    [13.81, 23.99, 67.39, 59.30]
]) / 10.0)
elec_cond_mixt_minus1 = (np.array([
    [8.364, 15.24, 39.84, 35.69],
    [8.526, 15.76, 40.61, 35.17],
    [8.433, 14.68, 40.21, 36.40]
]) / 10.0)
elec_cond_mixt_minus10 = (np.array([
    [6.665, 11.81, 32.22, 31.56],
    [6.665, 10.90, 33.13, 28.42],
    [6.665, 11.355, 33.31, 32.63]
]) / 10.0)

# Write CSV file
with open('Mahboub2026BenchtopData.csv', 'w', newline='') as csvfile:
    # Write header comments as plain text (not through csv.writer to avoid quoting issues)
    csvfile.write('# Mahboub et al. (2026) - Benchtop Conductivity Measurements\n')
    csvfile.write('#\n')
    csvfile.write('# Instrument: Thermo Scientific ORION Star A329 Conductivity Meter\n')
    csvfile.write('# Reference: See Mahboub et al. (2026) manuscript for full methodology\n')
    csvfile.write('#\n')
    csvfile.write('# Data description:\n')
    csvfile.write('# - All measurements from benchtop conductivity probe\n')
    csvfile.write('# - Gamry impedance data analyzed separately from raw Gamry files\n')
    csvfile.write('# - Three replicates per condition\n')
    csvfile.write('# - Temperatures: -10°C, -6°C, -3°C, -1°C, 5°C, 20°C, 25°C\n')
    csvfile.write('# - Compounds: NaCl, MgSO4, NH4Cl, Na2CO3, Mixture (1:1:1 MgSO4:NaCl:Na2CO3)\n')
    csvfile.write('#\n')
    csvfile.write('# Columns:\n')
    csvfile.write('#   compound: Chemical formula\n')
    csvfile.write('#   concentration_molal: Molality (mol/kg_H2O)\n')
    csvfile.write('#   temperature_C: Temperature (Celsius)\n')
    csvfile.write('#   temperature_K: Temperature (Kelvin)\n')
    csvfile.write('#   conductivity_Sm: Conductivity (S/m)\n')
    csvfile.write('#   source: Data source (\'benchtop\' for all entries)\n')
    csvfile.write('#   notes: Additional notes (e.g., original concentration in g/kg)\n')
    csvfile.write('#\n')

    # Now use csv.writer for the actual data
    writer = csv.writer(csvfile)
    writer.writerow(['compound', 'concentration_molal', 'temperature_C', 'temperature_K', 'conductivity_Sm', 'replicate', 'source', 'notes'])

    # NaCl data
    csvfile.write('# NaCl benchtop data - concentrations in mol/kg (molal), converted from g/kg by dividing by molar mass 58.44\n')
    nacl_data = [
        (elec_cond_nacl_25, 25),
        (elec_cond_nacl_20, 20),
        (elec_cond_nacl_5, 5),
        (elec_cond_nacl_minus1, -1),
        (elec_cond_nacl_minus3, -3),
        (elec_cond_nacl_minus6, -6),
        (elec_cond_nacl_minus10, -10)
    ]
    conc_nacl_gkg = np.array([10, 30, 50, 75, 100, 150])
    for data, temp_c in nacl_data:
        temp_k = temp_c + 273.15
        for rep in range(3):
            for i, conc in enumerate(conc_nacl):
                notes = f"{conc_nacl_gkg[i]} g/kg / 58.44" if rep == 0 else ""
                writer.writerow(['NaCl', f'{conc:.4f}', temp_c, temp_k, f'{data[rep, i]:.4g}', rep+1, 'benchtop', notes])

    # MgSO4 data
    csvfile.write('# MgSO4 benchtop data\n')
    mgso4_data = [
        (elec_cond_mgso4_25, 25),
        (elec_cond_mgso4_20, 20),
        (elec_cond_mgso4_5, 5),
        (elec_cond_mgso4_minus1, -1),
        (elec_cond_mgso4_minus3, -3),
        (elec_cond_mgso4_minus6, -6),
        (elec_cond_mgso4_minus10, -10)
    ]
    conc_mgso4_gkg = np.array([3, 40, 80, 120, 170, 200])
    for data, temp_c in mgso4_data:
        temp_k = temp_c + 273.15
        for rep in range(3):
            for i, conc in enumerate(conc_mgso4):
                if temp_c == -10 and data[rep, i] == 0.0:
                    notes = "Frozen (below eutectic point)" if rep == 0 else ""
                else:
                    notes = f"{conc_mgso4_gkg[i]} g/kg / 120.366" if rep == 0 else ""
                writer.writerow(['MgSO4', f'{conc:.4f}', temp_c, temp_k, f'{data[rep, i]:.4g}', rep+1, 'benchtop', notes])

    # NH4Cl data
    csvfile.write('# NH4Cl benchtop data\n')
    nh4cl_data = [
        (elec_cond_nh4cl_25, 25),
        (elec_cond_nh4cl_20, 20),
        (elec_cond_nh4cl_5, 5),
        (elec_cond_nh4cl_minus1, -1),
        (elec_cond_nh4cl_minus3, -3),
        (elec_cond_nh4cl_minus6, -6),
        (elec_cond_nh4cl_minus10, -10)
    ]
    conc_nh4cl_gkg = np.array([10, 50, 75, 100])
    for data, temp_c in nh4cl_data:
        temp_k = temp_c + 273.15
        for rep in range(3):
            for i, conc in enumerate(conc_nh4cl):
                notes = f"{conc_nh4cl_gkg[i]} g/kg / 53.491" if rep == 0 else ""
                writer.writerow(['NH4Cl', f'{conc:.4f}', temp_c, temp_k, f'{data[rep, i]:.4g}', rep+1, 'benchtop', notes])

    # Na2CO3 data
    csvfile.write('# Na2CO3 benchtop data\n')
    na2co3_data = [
        (elec_cond_na2co3_25, 25),
        (elec_cond_na2co3_20, 20),
        (elec_cond_na2co3_5, 5),
        (elec_cond_na2co3_minus1, -1),
        (elec_cond_na2co3_minus3, -3),
        (elec_cond_na2co3_minus6, -6),
        (elec_cond_na2co3_minus10, -10)
    ]
    conc_na2co3_gkg = np.array([5, 10, 25, 40, 55])
    for data, temp_c in na2co3_data:
        temp_k = temp_c + 273.15
        for rep in range(3):
            for i, conc in enumerate(conc_na2co3):
                notes = f"{conc_na2co3_gkg[i]} g/kg / 105.9888" if rep == 0 else ""
                writer.writerow(['Na2CO3', f'{conc:.4f}', temp_c, temp_k, f'{data[rep, i]:.4g}', rep+1, 'benchtop', notes])

    # Mixture data (1:1:1 MgSO4:NaCl:Na2CO3)
    csvfile.write('# Mixture data (1:1:1 MgSO4:NaCl:Na2CO3 by molality)\n')
    mixt_data = [
        (elec_cond_mixt_25, 25),
        (elec_cond_mixt_20, 20),
        (elec_cond_mixt_minus1, -1),
        (elec_cond_mixt_minus10, -10)
    ]
    for data, temp_c in mixt_data:
        temp_k = temp_c + 273.15
        for rep in range(3):
            for i, conc in enumerate(conc_mixt):
                writer.writerow(['Mixture', f'{conc:.2f}', temp_c, temp_k, f'{data[rep, i]:.4g}', rep+1, 'benchtop', ''])

print("CSV file generated successfully!")
print("\nFirst few NaCl entries (for verification):")
print("Concentration (molal), Temp (C), Conductivity (S/m)")
for i in range(6):
    print(f"{conc_nacl[i]:.4f}, 25, {elec_cond_nacl_25[0,i]:.4g}")
