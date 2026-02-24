import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import Blues
from matplotlib.colors import Normalize
from sigmaElectricMcCleskey2012 import elecCondMcCleskey2012

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# test for single-value input
# ions = {'Na_p1': {'mols': 0.1}, 'Cl_m1': {'mols': 0.1}}
# ionout = elecCondMcCleskey2012(25,ions)
# sigma_Sm = ionout['sigma_Sm'].item()
# print(f'sigma = {sigma_Sm:.4g} S/m')

def getDnames():
    return [
        ('K_p1', 39.0983, 3e-5, 11, 0.1, 0),
        ('Na_p1', 22.989769, 0.006, 470, 0.3, 1.63),
        ('Li_p1', 6.941, 0.001, 1.6, 0, 0),
        ('H_p1', 1.00784, 0, 0, 0, 0),
        ('Ca_p2', 40.078, 1e-3, 18, 0.1, 0.0064),
        ('Mg_p2', 24.305, 1e-5, 52, 1, 2.929),
        ('Ba_p2', 137.327, 0, 0, 0, 0),
        ('Sr_p2', 87.62, 1e-5, 0.22, 0, 0),
        ('Cl_m1', 35.453, 1e-3, 500, 0.25, 0.308),
        ('SO4_m2', 96.06, 6e-4, 570, 1, 3.5964),
        ('NO3_m1', 62.0049, 8e-4, 4.5, 0, 0),
        ('OH_m1', 17.008, 0, 0, 0, 0),
        ('CO3_m2', 60.01, 0, 0, 0, 0),
        ('HCO3_m1', 61.0168, 0.02, 14, 0, 0),
        ('F_m1', 18.998403, 3e-3, 2.9, 0, 0),
        ('Br_m1', 79.904, 6e-4, 0.79, 0, 0),
        ('Cs_p1', 132.90545, 0, 0, 0, 0),
        ('NH4_p1', 18.03846, 3e-3, 49, 0, 0),
        ('Mn_p2', 54.938049, 2e-6, 9.1, 0, 0),
        ('Zn_p2', 65.38, 2e-5, 19, 0, 0),
        ('Al_p3', 26.981538, 3e-5, 35, 0, 0),
        ('Cu_p2', 63.546, 2e-5, 3.5, 0, 0),
        ('Fe_p2', 55.845, 4e-5, 160, 0.001, 0),
        ('Fe_p3', 55.845, 4e-5, 100, 0, 0),
        ('KSO4_m1', 135.1609, 0, 0, 0, 0),
        ('NaSO4_m1', 119.05237, 0, 0, 0, 0),
        ('HSO4_m1', 97.07054, 0, 0, 0, 0),
        ('NaCO3_m1', 82.99867, 0, 0, 0, 0)
    ]

def format_ion_name(ion_name):
    formatted_name = ''
    i = 0
    while i < len(ion_name):
        char = ion_name[i]
        if char.isdigit():
            # Next character after a digit should be checked if it's still part of the subscript/superscript
            if i + 1 < len(ion_name) and ion_name[i + 1].isdigit():
                formatted_name += char
            else:
                formatted_name += f'_{char}'
        elif char in ('p', 'm'):
            # Assuming the next character defines the magnitude of the charge
            charge = '+' if char == 'p' else '-'
            if i + 1 < len(ion_name) and ion_name[i + 1].isdigit():
                charge += ion_name[i + 1]
                i += 1
            formatted_name += f'^{{{charge}}}'
        elif not char=='_':
            formatted_name += char
        i += 1
    return r'$' + formatted_name + r'$'


def plotStuff(mols, T_C, dnames, indices):
    # Set up the colormap and normalizer
    norm = Normalize(vmin=0, vmax=90)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
    scalar_map.set_array([])  # Ensures the scalar_map is recognized

    num_plots = len(indices)
    cols = 2
    rows = (num_plots + 1) // cols if num_plots % cols != 0 else num_plots // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, index in enumerate(indices):
        ax = axes[i // cols, i % cols]
        ion_name, _, _, _, _, _ = dnames[index]
        formatted_ion_name = format_ion_name(ion_name)
        ions = {ion_name: {'mols': mols}}
        ions = elecCondMcCleskey2012(T_C, ions)

        # Plot each temperature's data
        for temp_index, temp in enumerate(T_C):
            color = scalar_map.to_rgba(temp)
            ax.plot(mols, 0.1 * mols * ions[ion_name]['lamda'][temp_index], color=color)

        ax.set_xlabel('m (mol/kg)')
        ax.set_ylabel(r'$\sigma$ (S/m)')
        ax.set_title(f'Conductivity of {formatted_ion_name}', fontsize=12)
        ax.grid(True)

        # Add a color bar to each subplot
        cbar = fig.colorbar(scalar_map, ax=ax, orientation='vertical', label='Temperature (°C)')
        cbar.set_label('T (°C)')

    # Turn off unused axes
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    plt.tight_layout()
    plt.show()

    # # Normalizing the temperature values for colormap indexing
    # norm = Normalize(vmin=0, vmax=90)
    # scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=Blues)
    #
    # for i, index in enumerate(indices):
    #     ion_name, _, _, _, _, _ = dnames[index]
    #     ions = {ion_name: {'mols': mols}}
    #     ions = elecCondMcCleskey2012(T_C, ions)
    #
    #     plt.subplot(4, 2, i + 1)  # Adjust subplot indexing based on MATLAB's 1-based indexing
    #
    #     # Plot each temperature's data
    #     for i, temp in enumerate(T_C):
    #         color = scalar_map.to_rgba(temp)
    #         plt.plot(mols, 0.1 * mols * ions[ion_name]['lamda'][i], label=f'{ion_name} at {temp}°C', color=color)
    #
    #     # Adding color bar
    #     scalar_map.set_array([])
    #     cbar = plt.colorbar(scalar_map, orientation='vertical')
    #     cbar.set_label('Temperature (°C)')
    #
    #     plt.xlabel('m (mol/kg)')
    #     plt.ylabel(r'$\sigma$ (S/m)')
    #     plt.title(f'Conductivity of {ion_name}')
    #     plt.grid(True)
    #     # plt.legend()


def McCleskyFig1():
    mols = np.linspace(1e-4, 10 ** 0.6, 100)  # Assume 100 points for smooth curves
    T_C = [0, 5, 10, 25, 35, 45, 70, 90]
    dnames = getDnames()

    for i in range(0, len(dnames), 8):  # Assuming 8 plots per figure as example
        plt.figure(figsize=(10, 8))
        plotStuff(mols, T_C, dnames, range(i, min(i + 8, len(dnames))))
        plt.tight_layout()
        plt.show()


McCleskyFig1()
