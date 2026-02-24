from impedance import preprocessing
from impedance.models.circuits import Randles, CustomCircuit
import matplotlib.pyplot as plt
from impedance.visualization import plot_nyquist

import schemdraw
import schemdraw.elements as elm

import pandas as pd
import numpy as np
import sys
import logging
# Assign logger
log = logging.getLogger('HiPOZ')
stream = logging.StreamHandler(sys.stdout)
stream.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
log.setLevel(logging.DEBUG)
log.addHandler(stream)

# # Load data from the example EIS data
# frequencies, Z = preprocessing.readCSV('./data/2023091/ConductivityData_80mSKCltest/20230921-105603_Temp294K.txt')

# Define the file path for NaCl saturated
NaCl_file_path = 'data/20231214/ConductivityData_NaCl_4mol_kg/20231214-154247_Temp294K.txt'

# Define the file path for KCl standard
KCl_file_path = 'data/20231214/ConductivityData_KClStd_23us_cm/20231214-104516_Temp293K.txt'

# Read the data into a pandas DataFrame, skip the header lines, and specify the delimiter
data = pd.read_csv(NaCl_file_path, skiprows=9, delimiter='\t')

# Display the first few rows of the DataFrame to verify the data was loaded correctly
print(data.head())

# Convert the pandas Series to numpy arrays
frequencies = np.array(data['Frequency (Hz)'])
impedance = np.array(data['Impedance (ohm)'])
phase = np.array(data['Phase (degrees)'])

# Combine impedance and phase into a single complex column
Z = impedance * np.exp(1j * np.radians(phase))

# RC Circuit
R1 = r'R_1'
C1 = r'C_1'
RC_circuit = f'p({R1},{C1})'
Kest_pm = 80
sigmaStdCalc_Sm = 8
RC_initial_guess = [Kest_pm/sigmaStdCalc_Sm, 146.2e-12]

# Lima 2017 Circuit
# Define your circuit parameters
# Parameters
R0 = r'R_0'
R1 = r'R_1'
C1 = r'C_1'
C2 = r'C_2'
CPE1 = r'CPE_1'
L1 = r'L_1'
R2 = r'R_2'

# Define the custom circuit as a string
Lima_2017_circuit = f'p({C1},{R1})-p({C2},{CPE1})-{R2}-{L1}'
Lima_initial_guess = [1, 1, 1, 1, 1, 1, 1]
# Lima_initial_guess = [3e-6, 2e3, 3e-6, 4e-6, 0.75, 9, 4e-6]

# Lima_2017_circuit = f'R({R_b})-p(R({R_1}),C({C_1}))-s({A},p(R({R_2}),L({L_1})))'
# Z_1 = f'p({C1},{R1})'
# Z_2 = f'p({C2},{CPE1})'
# Z_3 = f's({R2}-{L1})'

# fit the CustomCircuit to the data
circuit = CustomCircuit(Lima_2017_circuit, initial_guess=(Lima_initial_guess))
circuit.fit(frequencies, Z)
log.info(circuit)

circFile = 'Lima2017Circuit.pdf'
Lleads = 1.6
with schemdraw.Drawing(file=circFile, show=False) as circ:
    circ.config(unit=Lleads)
    circ += elm.Line().length(circ.unit / 4).dot()
    circ += (j1 := elm.Line().length(circ.unit / 2).up())
    circ += elm.Line().at(j1.start).length(circ.unit / 2).down()
    circ += elm.Resistor().right().label(f'${R1}$')
    circ += elm.Line().length(circ.unit / 2).up().dot()
    circ += (j2 := elm.Line().length(circ.unit / 2).up())
    circ += elm.Capacitor().endpoints(j1.end, j2.end).label(f'${C1}$').right()
    circ += elm.Line().at(j2.start).length(circ.unit / 4).right().dot()

    circ += (j3 := elm.Line().length(circ.unit / 2).up())
    circ += elm.Line().at(j3.start).length(circ.unit / 2).down()
    circ += elm.CPE().right().label(f'$Z_\mathrm{{{CPE1[:-2]}}}$')
    circ += elm.Line().length(circ.unit / 2).up().dot()
    circ += (j4 := elm.Line().length(circ.unit / 2).up())
    circ += elm.Capacitor().endpoints(j3.end, j4.end).label(f'${C2}$').right()

    circ += elm.Resistor().at(j4.start).right().label(f'${R2}$')
    circ += elm.Inductor().right().label(f'${L1}$')
log.info(f'Equivalent circuit diagram saved to file: {circFile}')
