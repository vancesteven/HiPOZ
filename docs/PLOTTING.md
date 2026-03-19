# Generalized Study Plotting Functions

This document describes the generalized plotting system for conductivity studies, designed for Mahboub et al. (2026) and future publications (e.g., Cortes et al.).

## Overview

The plotting system has been refactored into four modules:

1. **`plotting.py`** - Low-level plotting functions (general purpose)
2. **`study_plots.py`** - High-level study-specific functions (reusable across studies)
3. **`config_plots.py`** - Universal plot configuration settings (fonts, colormaps, options)
4. **`mahboub2026_plots.py`** - Mahboub et al. (2026) specific plotting script

## Key Features

- **Automatic McCleskey model computation** - Computes deviation plots automatically
- **Gamry data overlays** - Automatically overlays impedance data on benchtop measurements
- **Delta subplots** - All concentration and temperature plots include Δ% deviation panels
- **Pressure plotting support** - New function for σ vs P plots following same format
- **CSV-based data storage** - Easy to edit in Excel, version-controllable
- **Configurable font sizes** - Single location to adjust all plot fonts
- **LaTeX formatting** - Publication-quality chemical formulas

## Module: `config_plots.py`

Universal configuration settings for all plotting scripts. Edit this file to change settings across all studies:

```python
# Font Sizes
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TITLE = 16
FONTSIZE_LEGEND = 10

# Colormaps (use 'tab10' for discrete colors, 'plasma'/'viridis' for continuous)
COLORMAP_CONCENTRATION = 'tab10'
COLORMAP_TEMPERATURE = 'tab10'
COLORMAP_PRESSURE = 'tab10'

# Plot Options
SHOW_LEGEND = False  # Include legend in plots
SHOW_TITLE = True    # Include title in plots
SHOW_DELTA = True    # Include Delta (Δ%) deviation subplots

# Output Settings
DPI = 300  # Resolution for saved plots
```

**Usage:**
```python
from config_plots import (
    FONTSIZE_AXIS_LABEL,
    COLORMAP_CONCENTRATION,
    SHOW_LEGEND
)
```

## Module: `study_plots.py`

### Data Loading

```python
from study_plots import load_study_data

# Load benchtop data from CSV
data = load_study_data('Mahboub2026BenchtopData.csv')
```

**CSV Format:**
```csv
compound,concentration_molal,temperature_C,temperature_K,conductivity_Sm,replicate,source,notes
NaCl,0.1711,25,298.15,1.788,1,benchtop,
NaCl,0.1711,25,298.15,1.777,2,benchtop,
...
```

### Plotting Functions

#### Concentration Plots

```python
from study_plots import plot_study_concentration

fig = plot_study_concentration(
    data=benchtop_data,
    compound='NaCl',
    output_file='nacl_vs_conc.pdf',
    gamry_data=None,  # Optional: (conc, sigma, errors, label) tuple
    show_delta=True,  # Include Δ% subplot
    show_legend=False,  # Hide legend (default)
    show_title=True,  # Show title (default)
    compound_latex=r'NaCl',  # LaTeX name for plot
    colormap='tab10',  # Discrete colormap (default)
    fontsize_label=14,
    fontsize_title=16,
    fontsize_legend=10
)
```

**Features:**
- Top panel: σ vs concentration with error bars
- Bottom panel: Δ% deviation from McCleskey model
- Automatic McCleskey model computation
- Optional Gamry data overlay (black-outlined markers)
- Color-coded by temperature

#### Temperature Plots

```python
from study_plots import plot_study_temperature

fig = plot_study_temperature(
    data=benchtop_data,
    compound='NaCl',
    output_file='nacl_vs_temp.pdf',
    show_delta=True,
    show_legend=False,  # Hide legend (default)
    show_title=True,  # Show title (default)
    compound_latex=r'NaCl',
    colormap='tab10',  # Discrete colormap (default)
    fontsize_label=14,
    fontsize_title=16,
    fontsize_legend=10
)
```

**Features:**
- Top panel: σ vs temperature with error bars
- Bottom panel: Δ% deviation from McCleskey model
- Color-coded by concentration
- Right-edge concentration labels

#### Pressure Plots (New!)

```python
from study_plots import plot_study_pressure

fig = plot_study_pressure(
    data=pressure_data,
    compound='NaCl',
    output_file='nacl_vs_pressure.pdf',
    show_delta=False,  # No pressure model yet
    compound_latex=r'NaCl',
    fontsize_label=14,
    fontsize_title=16,
    fontsize_legend=10
)
```

**Features:**
- Follows same format as temperature plots
- σ vs pressure with multiple concentration curves
- Ready for future model integration
- Optional Delta subplot when pressure model is available

### McCleskey Model Computation

```python
from study_plots import compute_mccleskey_model

# Compute model at specific points
model_sigmas = compute_mccleskey_model(
    conc_molal=[0.1, 0.5, 1.0],  # mol/kg
    temps_K=[273.15, 298.15, 323.15],  # K
    compound='NaCl'  # Or provide custom ion_spec
)
```

**Supported Compounds:**
- NaCl
- MgSO4
- NH4Cl
- Na2CO3

**Custom Ion Specification:**
```python
model_sigmas = compute_mccleskey_model(
    conc_molal=concs,
    temps_K=temps,
    ion_spec={'Na_p1': 2.0, 'SO4_m2': 1.0}  # Na2SO4
)
```

## Mahboub et al. (2026) Implementation

### File Structure

```
Mahboub2026BenchtopData.csv  # All benchtop measurements + Gamry + McCleskey reference
mahboub2026_plots.py          # Plotting script (generates 10 plots)
mahboub_plots/                # Output directory
  ├── nacl_vs_concentration.pdf
  ├── nacl_vs_temperature.pdf
  ├── mgso4_vs_concentration.pdf
  ├── ... (10 plots total)
```

### Running

```bash
python mahboub2026_plots.py
```

Generates:
- 5 compounds × 2 plot types = 10 plots
- All plots include Delta subplots (except Mixture - no model)
- Gamry overlays on NaCl and MgSO4
- Configurable font sizes at top of file

### Font Size Configuration

Edit three variables at top of `mahboub2026_plots.py`:

```python
FONTSIZE_AXIS_LABEL = 14  # Axis labels
FONTSIZE_TITLE = 16        # Plot titles
FONTSIZE_LEGEND = 10       # Legend and right-edge labels
```

## Data Sources

### Benchtop Probe

**Instrument:** Thermo Scientific ORION Star A329 Conductimeter

**Error Budget:**
- Probe calibration: 0.5%
- Temperature control: 2.89%
- Concentration preparation: 0.5%
- Combined: σ_total = √(σ_random² + 0.005² + 0.0289² + 0.005²)

### Gamry Impedance

**Processing:** HiPOZ impedance analysis pipeline
- Circuit fitting uncertainties
- Cell constant uncertainties
- Typical: 4-7% combined uncertainty

### McCleskey Model

**Reference:** McCleskey et al. (2012)
- Empirical conductivity model for electrolytes
- Valid ranges compound-specific
- NaCl: up to ~1 mol/kg
- MgSO4: up to ~0.012 mol/kg
- See original paper for full ranges

## Creating New Studies

### Step 1: Create CSV File

Follow Mahboub2026BenchtopData.csv format:
- Include all metadata (temperature, concentration, replicate, source)
- Use comment lines (starting with #) for documentation
- Consistent units: mol/kg, K, S/m

### Step 2: Create Plotting Script

```python
#!/usr/bin/env python3
"""Cortes et al. (2027) - Conductivity Plots"""

import os
from study_plots import (
    load_study_data,
    plot_study_concentration,
    plot_study_temperature,
    plot_study_pressure  # New!
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

# Study-specific configuration
OUTPUT_DIR = 'cortes_plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
data = load_study_data('Cortes2027Data.csv')

# Generate plots
print("Generating plots...")

# Concentration plot
plot_study_concentration(
    data=data,
    compound='KCl',
    output_file=os.path.join(OUTPUT_DIR, 'kcl_vs_conc.pdf'),
    show_delta=True,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

# Temperature plot
plot_study_temperature(
    data=data,
    compound='KCl',
    output_file=os.path.join(OUTPUT_DIR, 'kcl_vs_temp.pdf'),
    show_delta=True,
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

# Pressure plot (new!)
plot_study_pressure(
    data=data,
    compound='KCl',
    output_file=os.path.join(OUTPUT_DIR, 'kcl_vs_pressure.pdf'),
    show_delta=False,  # No pressure model yet
    fontsize_label=FONTSIZE_AXIS_LABEL,
    fontsize_title=FONTSIZE_TITLE,
    fontsize_legend=FONTSIZE_LEGEND
)

print("All plots saved!")
```

### Step 3: Run

```bash
python cortes2027_plots.py
```

## Extending for Pressure Data

To add pressure measurements to CSV:

```csv
compound,concentration_molal,temperature_C,temperature_K,pressure_MPa,conductivity_Sm,replicate,source,notes
NaCl,1.0,25,298.15,0.1,8.5,1,benchtop,Atmospheric pressure
NaCl,1.0,25,298.15,10,8.7,1,pressure_cell,10 MPa
NaCl,1.0,25,298.15,50,9.2,1,pressure_cell,50 MPa
...
```

Then use `plot_study_pressure()` to generate σ vs P plots following the same format as temperature plots.

## Advanced: Custom Ion Specifications

For compounds not in the default list:

```python
from study_plots import ION_SPECS

# Add custom compound
ION_SPECS['KCl'] = {'K_p1': 1.0, 'Cl_m1': 1.0}

# Or pass directly to compute_mccleskey_model
model = compute_mccleskey_model(
    conc_molal=concs,
    temps_K=temps,
    ion_spec={'K_p1': 1.0, 'Cl_m1': 1.0}
)
```

## Summary

**Benefits of generalized system:**
- 90% code reuse across studies
- Consistent plot formatting
- Easy to add new compounds
- Pressure plotting ready for Cortes et al.
- Delta subplots automatic
- Gamry overlay support built-in

**For Cortes et al.:**
- Copy Mahboub CSV structure
- Create new plotting script
- Add pressure measurements
- Use `plot_study_pressure()` for P-dependence
- All other functions work out of the box!
