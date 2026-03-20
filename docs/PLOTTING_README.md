# HiPOZ Plotting Guide

Comprehensive guide to plotting conductivity data in HiPOZ, including both Gamry impedance measurements and external benchtop probe data.

## Overview

The new `plotting.py` module provides flexible plotting functions for:

1. **Conductivity vs Concentration** - Multiple temperature curves
2. **Conductivity vs Temperature** - Multiple concentration curves
3. **Conductivity vs Pressure** - Gamry measurements colored by temperature
4. **Nyquist Plots** - Complex impedance diagrams
5. **Bode Plots** - Magnitude and phase vs frequency
6. **External Data Integration** - Benchtop probe data from CSV/text files

## Quick Start

### For Gamry Data (GUI Mode)

```bash
python gamry_HiPOZ.py --dates 20250815
```

The GUI now includes buttons to generate σ vs concentration and σ vs temperature plots.

### For External Data

```python
from plotting import plot_sigma_vs_concentration

# Your data arrays
concentrations = [0.5, 1.0, 1.5]  # mol/kg
conductivities_at_25C = [4.5, 8.2, 11.5]  # S/m
conductivities_at_20C = [4.1, 7.8, 11.0]  # S/m

plot_sigma_vs_concentration(
    conc_data=concentrations,
    sigma_data=[conductivities_at_20C, conductivities_at_25C],
    temp_labels=['293.15 K (20°C)', '298.15 K (25°C)'],
    out_file='my_data.pdf'
)
```

## Command Line Options

### New Plotting Flags

```bash
# Conductivity vs concentration
python gamry_HiPOZ.py --headless --dates 20250815 --plot-sigma-conc

# Conductivity vs temperature
python gamry_HiPOZ.py --headless --dates 20250815 --plot-sigma-temp

# All plots including new ones
python gamry_HiPOZ.py --headless --dates 20250815 --plot-all

# Combine multiple plot types
python gamry_HiPOZ.py --headless --dates 20250815 --plot-sigma-conc --plot-sigma-temp --plot-svsp
```

### Existing Plotting Flags

```bash
--plot-svsp       # Conductivity vs pressure
--plot-bode       # Bode plots (|Z| and phase vs frequency)
--plot-nyquist    # Nyquist plots (Re(Z) vs -Im(Z))
--plot-all        # Generate all available plots
```

## Function Reference

### Core Plotting Functions

#### `plot_sigma_vs_concentration()`

Plot conductivity vs concentration with multiple temperature curves.

**Parameters:**
- `conc_data` (array): Concentration values (e.g., mol/kg)
- `sigma_data` (list of arrays): Conductivity at each temperature
- `temp_labels` (list of str, optional): Temperature labels
- `sigma_errors` (list of arrays, optional): Error bars
- `model_data` (list of arrays, optional): Model predictions (dashed lines)
- `xlabel` (str): X-axis label (default: mol/kg)
- `title` (str): Plot title
- `show_delta` (bool): Show percent difference panel if model provided
- `limit` (float, optional): Vertical line marking validity limit
- `out_file` (str, optional): Save to file
- `colormap` (str): Matplotlib colormap (default: 'tab10')

**Returns:**
- `fig`: Matplotlib figure object

**Example:**
```python
from plotting import plot_sigma_vs_concentration
import numpy as np

# NaCl data
conc = np.array([0.17, 0.51, 0.86, 1.28, 1.71, 2.57])  # mol/kg
sigma_25C = np.array([1.78, 4.86, 7.49, 10.57, 13.23, 17.66])  # S/m
sigma_20C = np.array([1.62, 4.42, 6.81, 9.61, 12.03, 16.05])  # S/m
sigma_5C = np.array([1.19, 3.12, 4.79, 6.81, 8.60, 11.28])  # S/m

fig = plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=[sigma_5C, sigma_20C, sigma_25C],
    temp_labels=['278.15 K', '293.15 K', '298.15 K'],
    title='NaCl Conductivity',
    out_file='NaCl_vs_conc.pdf'
)
```

#### `plot_sigma_vs_temperature()`

Plot conductivity vs temperature with multiple concentration curves.

**Parameters:**
- `temp_data` (array): Temperature values (K)
- `sigma_data` (list of arrays): Conductivity at each concentration
- `conc_labels` (list of str, optional): Concentration labels
- `sigma_errors` (list of arrays, optional): Error bars
- `model_data` (list of arrays, optional): Model predictions (dashed lines)
- `xlabel` (str): X-axis label (default: Temperature (K))
- `title` (str): Plot title
- `show_delta` (bool): Show percent difference panel if model provided
- `out_file` (str, optional): Save to file
- `colormap` (str): Matplotlib colormap (default: 'tab10')

**Returns:**
- `fig`: Matplotlib figure object

**Example:**
```python
from plotting import plot_sigma_vs_temperature
import numpy as np

# MgSO4 data
temps = np.array([263, 267, 270, 272, 278, 293, 298])  # K
sigma_0p5M = np.array([2.2, 2.5, 2.8, 3.0, 3.5, 4.8, 5.2])  # S/m
sigma_1p0M = np.array([4.0, 4.5, 5.0, 5.5, 6.2, 8.5, 9.5])  # S/m

fig = plot_sigma_vs_temperature(
    temp_data=temps,
    sigma_data=[sigma_0p5M, sigma_1p0M],
    conc_labels=['0.5 mol/kg', '1.0 mol/kg'],
    title='MgSO₄ Conductivity',
    out_file='MgSO4_vs_temp.pdf'
)
```

#### `plot_nyquist()`

Plot Nyquist diagram (Re(Z) vs -Im(Z)) for Gamry impedance data.

**Parameters:**
- `solutions` (list): List of Solution objects with Z_ohm attribute
- `out_file` (str, optional): Output file path
- `title` (str): Plot title
- `show_fit` (bool): Show circuit fit curves (default: True)
- `colormap` (str): Matplotlib colormap

**Example:**
```python
from plotting import plot_nyquist
from gamryTools import Solution

# Load Gamry data
solutions = [...]  # Your Solution objects

plot_nyquist(solutions, out_file='nyquist.pdf', title='Impedance Spectra')
```

#### `plot_bode()`

Plot Bode diagram (|Z| and phase vs frequency).

**Parameters:**
- `solutions` (list): List of Solution objects
- `out_file` (str, optional): Output file path
- `title` (str): Plot title
- `show_fit` (bool): Show circuit fit curves (default: True)
- `colormap` (str): Matplotlib colormap

**Example:**
```python
from plotting import plot_bode

plot_bode(solutions, out_file='bode.pdf')
```

#### `plot_conductivity_vs_pressure()`

Plot conductivity vs pressure for Gamry measurements.

**Parameters:**
- `measurements` (list): List of Solution objects with P_MPa, sigma_Sm, T_K
- `out_file` (str, optional): Output file path
- `title` (str): Plot title
- `color_by` (str): 'temperature' or 'concentration'

**Example:**
```python
from plotting import plot_conductivity_vs_pressure

plot_conductivity_vs_pressure(
    measurements,
    out_file='sigma_vs_P.pdf',
    color_by='temperature'
)
```

### Helper Functions

#### `safe_errorbar()`

Handle mismatched array sizes when plotting error bars.

#### `pdiff()`

Calculate percent difference: 100*(x - y)/y with NaN-safe behavior.

#### `tight_xlim()`

Set x-axis limits snug to data with optional padding.

#### `annotate_right()`

Add non-overlapping labels on right side of plot (like MahboubEtAl2026.py).

### Data Loading Functions

#### `load_external_data()`

Load conductivity data from external files.

**Parameters:**
- `file_path` (str): Path to data file
- `format` (str): 'csv', 'txt', or 'npy'

**Returns:**
- `data` (dict): Dictionary with keys 'concentration', 'temperature', 'conductivity', 'errors'

**CSV Format:**
```csv
concentration,temperature,conductivity,error
0.5,273.15,4.5,0.2
1.0,273.15,8.2,0.3
0.5,293.15,5.1,0.2
1.0,293.15,9.5,0.4
```

**Example:**
```python
from plotting import load_external_data, plot_sigma_vs_concentration
import numpy as np

# Load data
data = load_external_data('probe_data.csv', format='csv')

# Group by temperature
temps = np.unique(data['temperature'])
sigma_by_temp = []

for T in temps:
    mask = data['temperature'] == T
    sigma_by_temp.append(data['conductivity'][mask])

# Plot
conc = data['concentration'][data['temperature'] == temps[0]]
plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_by_temp,
    temp_labels=[f'{T:.1f} K' for T in temps],
    out_file='loaded_data.pdf'
)
```

## Complete Example Scripts

### Example 1: NaCl Benchtop Data

See `plot_external_data_example.py`:

```bash
python plot_external_data_example.py
```

This script demonstrates:
- Plotting multiple temperatures vs concentration
- Error bar handling
- Label formatting
- PDF output

### Example 2: Comparing Gamry and Benchtop Data

```python
from gamryTools import Solution, TimeSeries
from plotting import plot_sigma_vs_concentration
import numpy as np

# Load Gamry data
gamry_data = [...]  # Your Solution objects
gamry_concs = np.array([s.w_molal for s in gamry_data])
gamry_sigmas = np.array([s.sigma_Sm for s in gamry_data])

# External benchtop data
bench_concs = np.array([0.5, 1.0, 1.5])
bench_sigmas = np.array([4.5, 8.2, 11.5])

# Plot both
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(bench_concs, bench_sigmas, 'o-', label='Benchtop probe', linewidth=2)
ax.plot(gamry_concs, gamry_sigmas, 's--', label='Gamry impedance', linewidth=2)
ax.set_xlabel('Concentration (mol/kg)', fontsize=14)
ax.set_ylabel('Conductivity (S/m)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.savefig('gamry_vs_benchtop.pdf', dpi=300)
```

### Example 3: McCleskey Model Comparison

```python
from plotting import plot_sigma_vs_concentration
import numpy as np
import importlib.util

# Load McCleskey model
spec = importlib.util.spec_from_file_location(
    "sigmaElectricMcCleskey2012",
    "sigmaElectricMcCleskey2012.py"
)
mc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc_mod)

# Experimental data
conc = np.array([0.1, 0.5, 1.0])  # mol/kg
sigma_exp_25C = np.array([1.5, 7.5, 14.5])  # S/m
sigma_err_25C = np.array([0.05, 0.3, 0.5])  # S/m

# Model predictions
ions = {"Na_p1": {"mols": conc}, "Cl_m1": {"mols": conc}}
T_vec = np.full_like(conc, 25.0)  # 25°C
sigma_model_25C = mc_mod.elecCondMcCleskey2012(0.1, T_vec, ions)

# Plot with model comparison
fig = plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=[sigma_exp_25C],
    sigma_errors=[sigma_err_25C],
    model_data=[sigma_model_25C],
    temp_labels=['298.15 K (25°C)'],
    show_delta=True,  # Shows percent difference panel
    title='NaCl: Experiment vs McCleskey Model',
    out_file='nacl_model_comparison.pdf'
)
```

## Integration with HiPOZ Workflow

### Step 1: Measure with Gamry

```bash
# Process Gamry data
python gamry_HiPOZ.py --dates 20250815

# Mark standards and associate measurements in GUI
# GUI auto-saves to zAnalysis20250815.csv
```

### Step 2: Plot Gamry Results

```bash
# Generate all Gamry plots
python gamry_HiPOZ.py --headless --dates 20250815 --plot-all
```

### Step 3: Add External Comparison Data

```python
# Load your Gamry results
import pandas as pd
gamry_results = pd.read_csv('data/20250815/hipoz_results.csv')

# Load benchtop data
from plotting import load_external_data
bench_data = load_external_data('benchtop_probe_data.csv')

# Plot comparison (see Example 2 above)
```

## Matplotlib Configuration

The plotting module automatically configures Matplotlib for publication-quality output:

```python
plt.rcParams["pdf.fonttype"] = 42     # TrueType fonts (Illustrator-editable)
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"  # Keep SVG text as text
```

If LaTeX is available, it will be used for beautiful math typesetting:
```python
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{stix}...'
```

If LaTeX is not available, standard Matplotlib text rendering is used (no errors).

## Colormaps

Available colormaps (specify with `colormap` parameter):

- `'tab10'` - Default, 10 distinct colors
- `'viridis'` - Perceptually uniform, good for temperature
- `'plasma'` - Good for concentration gradients
- `'Set3'` - Pastel colors
- `'coolwarm'` - Blue-red diverging
- Any Matplotlib colormap name

## Tips and Best Practices

### Error Bars

Always provide error estimates when available:
```python
plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_means,
    sigma_errors=sigma_uncertainties,  # Include these!
    ...
)
```

### Multiple Temperatures

Order temperature curves from coldest to warmest for better readability:
```python
temps = ['263.15 K', '273.15 K', '298.15 K']  # Cold to warm
```

### Concentration Units

Be consistent with units. The module supports:
- `w_molal`: mol/kg_H2O (molality) - recommended for comparison with models
- `w_ppt`: g/kg_solution (parts per thousand) - convenient for preparation

Convert between them:
```python
# g/kg → mol/kg
molal = (ppt / molar_mass) / (1 + ppt/1000)

# mol/kg → g/kg
ppt = (molal * molar_mass) / (1 + (molal * molar_mass)/1000)
```

### File Formats

The plotting module supports:
- **PDF** - Vector graphics, best for publications (recommended)
- **PNG** - Raster, good for presentations (use dpi=300)
- **SVG** - Vector graphics, editable in Inkscape

```python
plot_sigma_vs_concentration(..., out_file='figure.pdf')
```

## Troubleshooting

### LaTeX Errors

If you see LaTeX-related errors:
```
! LaTeX Error: File 'stix.sty' not found
```

The module will fall back to standard text. To use LaTeX rendering:
1. Install a TeX distribution (TeX Live, MiKTeX, MacTeX)
2. Install required packages: stix, siunitx, upgreek, mhchem

Or disable LaTeX in your script:
```python
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
```

### Data Size Mismatches

If you see errors about array size mismatches:
```python
# Use safe_errorbar instead of ax.errorbar
from plotting import safe_errorbar
safe_errorbar(ax, x, y, yerr=errors)
```

### Empty Plots

Check that your data contains finite values:
```python
import numpy as np
data = np.array(data)
data = data[np.isfinite(data)]  # Remove NaN and inf
```

## Citation

If you use these plotting functions in published work, please cite:

- HiPOZ: [Citation info for HiPOZ]
- Impedance.py: https://impedancepy.readthedocs.io
- Matplotlib: Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.

## Support

For questions or issues:
- GitHub Issues: https://github.com/[your-repo]/issues
- Email: steven.d.vance@jpl.nasa.gov

## See Also

- `CLAUDE.md` - Developer guide
- `README.md` - Main project documentation
- `CALIBRATION_WORKFLOW.md` - Calibration procedures
- `MahboubEtAl2026.py` - Example analysis workflow
