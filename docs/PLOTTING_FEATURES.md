# HiPOZ Plotting Features

## Overview

The HiPOZ plotting system (`plotting.py`) provides publication-quality visualization functions for conductivity data. This document describes the main plotting functions and their usage.

## Core Plotting Functions

### 1. Conductivity vs Concentration (σ vs m)

**Function:** `plot_sigma_vs_concentration()`

**Purpose:** Plot conductivity as a function of molality at multiple temperatures.

**Location:** `plotting.py:244`

#### Parameters

```python
plot_sigma_vs_concentration(
    conc_data,           # array-like (N,): Concentration values (mol/kg)
    sigma_data,          # list of arrays (M×N): Conductivity at each temperature
    temp_labels=None,    # list of str (M,): Temperature labels (e.g., ['263 K', '273 K'])
    sigma_errors=None,   # list of arrays (M×N): Error bars for each temperature
    model_data=None,     # list of arrays (M×N): Model predictions (dashed lines)
    xlabel=r'Concentration (mol/kg$_{\mathrm{H_2O}}$)',
    title='Conductivity vs Concentration',
    show_delta=False,    # Show percent difference panel below main plot
    show_title=True,
    limit=None,          # Vertical line marking McCleskey validity limit
    out_file=None,       # Save to file
    colormap='tab10',
    fontsize_label=None,
    fontsize_title=None,
    fontsize_legend=None
)
```

#### Usage Example

```python
import numpy as np
from plotting import plot_sigma_vs_concentration

# Concentration range
conc = np.array([0.5, 1.0, 1.5, 2.0])  # mol/kg

# Conductivity data at three temperatures
sigma_263K = np.array([5.2, 9.8, 13.5, 16.2])  # S/m
sigma_273K = np.array([6.1, 11.5, 15.8, 19.0])
sigma_283K = np.array([7.0, 13.2, 18.1, 21.8])

sigma_data = [sigma_263K, sigma_273K, sigma_283K]
temp_labels = ['263 K', '273 K', '283 K']

# Optional error bars (5% uncertainty)
errors = [s * 0.05 for s in sigma_data]

# Create plot
fig = plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_data,
    temp_labels=temp_labels,
    sigma_errors=errors,
    title='NaCl Conductivity vs Concentration',
    out_file='nacl_sigma_vs_conc.pdf'
)
```

#### Plot Features

**Main Panel:**
- Markers (circles) show experimental data
- No connecting lines (discrete measurements)
- Error bars with caps
- Color-coded by temperature
- Right-edge labels identify curves
- Grid for readability
- Y-axis starts at 0 (conductivity ≥ 0)

**Model Comparison:**
```python
# Add McCleskey model overlay
model_predictions = [
    mccleskey_model(conc, T=263.15),
    mccleskey_model(conc, T=273.15),
    mccleskey_model(conc, T=283.15)
]

fig = plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_data,
    temp_labels=temp_labels,
    model_data=model_predictions,  # Dashed lines
    show_delta=True,                # Add percent difference panel
    limit=6.0                       # McCleskey validity limit
)
```

**Delta Panel (optional):**
- Below main plot when `show_delta=True`
- Shows percent difference: Δ = 100 × (σ_exp - σ_model) / σ_model
- Same color scheme as main plot
- Fixed ±5% error bars on differences
- Separate x-axis for independent zooming

**McCleskey Validity Limit:**
```python
# Add vertical line at 6.0 mol/kg (McCleskey limit for NaCl)
fig = plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_data,
    limit=6.0  # Dashed red line with label
)
```

#### Customization

**Font Sizes:**
```python
fig = plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_data,
    fontsize_label=14,   # Axis labels
    fontsize_title=16,   # Title
    fontsize_legend=12   # Right-edge labels
)
```

**Colormap:**
```python
# Qualitative (discrete colors)
colormap='tab10'      # Default: 10 distinct colors
colormap='Set1'       # 9 distinct colors
colormap='Paired'     # 12 paired colors

# Sequential (continuous gradient)
colormap='plasma'     # Purple to yellow
colormap='viridis'    # Blue to yellow
colormap='coolwarm'   # Blue to red
```

**X-axis Label:**
```python
# Molality (default)
xlabel=r'Concentration (mol/kg$_{\mathrm{H_2O}}$)'

# Parts per thousand
xlabel=r'Concentration (g/kg)'

# Mole fraction
xlabel=r'$x_{\mathrm{NaCl}}$'
```

### 2. Conductivity vs Temperature (σ vs T)

**Function:** `plot_sigma_vs_temperature()`

**Purpose:** Plot conductivity as a function of temperature at multiple concentrations.

**Location:** `plotting.py:380`

#### Parameters

```python
plot_sigma_vs_temperature(
    temp_data,           # array-like (N,): Temperature values (K)
    sigma_data,          # list of arrays (M×N): Conductivity at each concentration
    conc_labels=None,    # list of str (M,): Concentration labels (e.g., ['0.5 mol/kg'])
    sigma_errors=None,   # list of arrays (M×N): Error bars for each concentration
    model_data=None,     # list of arrays (M×N): Model predictions (dashed lines)
    xlabel=r'Temperature (K)',
    title='Conductivity vs Temperature',
    show_delta=False,    # Show percent difference panel below main plot
    show_title=True,
    out_file=None,
    colormap='tab10',
    fontsize_label=None,
    fontsize_title=None,
    fontsize_legend=None
)
```

#### Usage Example

```python
from plotting import plot_sigma_vs_temperature

# Temperature range
temps = np.array([263.15, 273.15, 283.15, 293.15])  # K

# Conductivity at three concentrations
sigma_05m = np.array([5.2, 6.1, 7.0, 7.9])  # 0.5 mol/kg
sigma_10m = np.array([9.8, 11.5, 13.2, 14.9])  # 1.0 mol/kg
sigma_15m = np.array([13.5, 15.8, 18.1, 20.4])  # 1.5 mol/kg

sigma_data = [sigma_05m, sigma_10m, sigma_15m]
conc_labels = ['0.5 mol/kg', '1.0 mol/kg', '1.5 mol/kg']

# Create plot
fig = plot_sigma_vs_temperature(
    temp_data=temps,
    sigma_data=sigma_data,
    conc_labels=conc_labels,
    title='NaCl Conductivity vs Temperature',
    out_file='nacl_sigma_vs_temp.pdf'
)
```

#### Plot Features

**Main Panel:**
- Markers (circles) show experimental data
- No connecting lines (discrete measurements)
- Error bars with caps
- Color-coded by concentration
- Right-edge labels identify curves
- Grid for readability
- Y-axis starts at 0
- X-axis range auto-adjusted with 10% padding on right

**Model Comparison:**
```python
# Add model overlay and comparison panel
model_predictions = [
    mccleskey_model(temps, m=0.5),
    mccleskey_model(temps, m=1.0),
    mccleskey_model(temps, m=1.5)
]

fig = plot_sigma_vs_temperature(
    temp_data=temps,
    sigma_data=sigma_data,
    model_data=model_predictions,
    show_delta=True  # Add percent difference panel
)
```

**Delta Panel:**
- Same behavior as σ vs m plots
- Useful for assessing model accuracy across temperature range

#### Typical Use Cases

**Study Temperature Dependence:**
```python
# Examine how σ changes with T at fixed composition
fig = plot_sigma_vs_temperature(
    temp_data=temps,
    sigma_data=[sigma_05m, sigma_10m, sigma_15m],
    conc_labels=['0.5 mol/kg', '1.0 mol/kg', '1.5 mol/kg'],
    title='Temperature Dependence of NaCl Conductivity'
)
```

**Compare to Literature:**
```python
# Overlay literature data for validation
lit_temps = np.array([273.15, 293.15])
lit_sigma = np.array([11.5, 14.9])  # Literature values at 1.0 mol/kg

fig = plot_sigma_vs_temperature(
    temp_data=temps,
    sigma_data=[sigma_10m],
    conc_labels=['This work'],
    model_data=[lit_sigma],  # Will interpolate/show as dashed line
    show_delta=True
)
```

### 3. Conductivity vs Pressure (σ vs P)

**Function:** `plot_conductivity_vs_pressure()`

**Purpose:** Scatter plot of conductivity vs pressure, color-coded by temperature.

**Location:** `plotting.py:620`

#### Parameters

```python
plot_conductivity_vs_pressure(
    measurements,        # list of Solution objects
    out_file=None,
    title='Conductivity vs Pressure',
    show_legend=True,
    temp_colormap='coolwarm'
)
```

#### Usage Example

```python
from plotting import plot_conductivity_vs_pressure

# measurements is list of Solution objects from gamry_HiPOZ.py
# Each has: P_MPa, conductivity_Sm, T_K, conductivity_unc_pct

fig = plot_conductivity_vs_pressure(
    measurements=calibrated_solutions,
    title='NaCl Conductivity vs Pressure',
    out_file='sigma_vs_p.pdf'
)
```

#### Plot Features

- **Scatter plot:** One point per measurement
- **Color coding:** Temperature gradient (coolwarm colormap)
- **Error bars:** Propagated uncertainty from calibration
- **Colorbar:** Shows temperature scale
- **Grid:** For reading values
- **Automatic scaling:** Adjusts to data range

### 4. Impedance Plots

#### Nyquist Plot

**Function:** `plot_nyquist()`

**Purpose:** Plot -Im(Z) vs Re(Z) for impedance spectra.

**Location:** `plotting.py:504`

```python
from plotting import plot_nyquist

fig = plot_nyquist(
    solutions=[sol1, sol2, sol3],  # List of Solution objects
    out_file='nyquist.pdf',
    title='Nyquist Plot - NaCl Solutions'
)
```

**Features:**
- Semicircle indicates parallel RC behavior
- Deviations show electrode effects (CPE)
- Circuit fit overlays (if available)
- Color-coded by solution

#### Bode Plot

**Function:** `plot_bode()`

**Purpose:** Plot |Z| and phase vs frequency.

**Location:** `plotting.py:557`

```python
from plotting import plot_bode

fig = plot_bode(
    solutions=[sol1, sol2, sol3],
    out_file='bode.pdf',
    title='Bode Plot - NaCl Solutions'
)
```

**Features:**
- Two panels: magnitude and phase
- Log-log scale for magnitude
- Semi-log for phase
- Circuit fit overlays
- Frequency range: 10 Hz - 1 MHz typical

## LaTeX Table Generation

### Purpose

Generate publication-ready tables following journal supplement formatting.

**Script:** `generate_cortes_latex_tables.py`

### Features

#### 1. Uncertainty Formatting

```python
from generate_cortes_latex_tables import format_uncertainty

# Format value ± uncertainty
result = format_uncertainty(
    value=12.34,
    uncertainty=0.56,
    decimals=2
)
# Returns: "12.34 $\\pm$ 0.56"

# Handle missing uncertainty
result = format_uncertainty(
    value=12.34,
    uncertainty=np.nan,
    decimals=2
)
# Returns: "12.34"
```

#### 2. Generate Tables

```python
import pandas as pd
from generate_cortes_latex_tables import generate_nacl_table

# Load data
data = pd.read_csv('cortes_data.csv')

# Generate LaTeX table
generate_nacl_table(
    data=data,
    output_file='cortes_nacl_table.tex'
)
```

**Output Format:**
```latex
\begin{table}[ht]
\centering
\begin{tabular}{lrrrrr}
\hline
$w$ (molal) NaCl & $T$ (K) & $P$ (MPa) & $Z$ (\si{\ohm}) & $\sigma$ (\si{S/m}) \\
\hline
Standard & 273 & 0.1 & 119.05 $\pm$ 0.60 & 0.0084 $\pm$ 0.0004 \\
1.5 & 273 & 126 & 8.42 $\pm$ 0.42 & 11.88 $\pm$ 0.59 \\
\hline
\end{tabular}
\caption{NaCl conductivity data}
\label{tab:nacl}
\end{table}
```

#### 3. Table Structure

**Columns:**
- Concentration (molal or ppt)
- Temperature (K)
- Pressure (MPa)
- Impedance (Ω) with uncertainty
- Conductivity (S/m) with uncertainty

**Rows:**
- Standards labeled explicitly
- Sorted by concentration, then T, then P
- Replicates shown together
- Excluded files omitted

#### 4. siunitx Integration

Uses LaTeX siunitx package for units:
```latex
\si{\ohm}     % Ohm symbol
\si{S/m}      % Siemens per meter
\si{MPa}      % Megapascal
\si{K}        % Kelvin
```

**Required Preamble:**
```latex
\usepackage{siunitx}
\usepackage{booktabs}  % For \toprule, \midrule, \bottomrule
```

#### 5. Multi-compound Tables

```python
# For multi-component solutions
generate_multicomp_table(
    data=data,
    compounds=['NaCl', 'MgSO4'],
    output_file='ocean_analog_table.tex'
)
```

**Output:**
```latex
$w$ (molal) NaCl & $w$ (molal) MgSO$_4$ & $T$ (K) & $P$ (MPa) & ...
1.5 & 0.6 & 273 & 126 & ... \\
```

### Usage in Publications

**Workflow:**
1. Run HiPOZ analysis to generate curated CSV
2. Run `generate_cortes_latex_tables.py` on CSV
3. Include `.tex` file in LaTeX manuscript:
   ```latex
   \input{cortes_nacl_table.tex}
   ```
4. Compile with pdflatex or xelatex

**Customization:**
- Edit `generate_cortes_latex_tables.py` for journal-specific formatting
- Adjust decimal places via `decimals` parameter
- Modify column order in `\begin{tabular}` line
- Add footnotes, caption, label as needed

## Helper Functions

### 1. Safe Error Bars

**Function:** `safe_errorbar()`

**Purpose:** Plot error bars that handle NaN and zero values gracefully.

```python
from plotting import safe_errorbar

safe_errorbar(
    ax, x, y,
    yerr=errors,
    fmt='o',
    color='blue',
    ms=8,        # Marker size
    mew=1.5,     # Marker edge width
    capsize=4,   # Error bar cap size
    label='Data'
)
```

**Features:**
- Filters out NaN values automatically
- Skips error bars for zero uncertainties
- Prevents matplotlib warnings
- Consistent marker styling

### 2. Percent Difference

**Function:** `pdiff()`

**Purpose:** Calculate percent difference for model comparison.

```python
from plotting import pdiff

delta = pdiff(experimental, model)
# Returns: 100 * (experimental - model) / model
```

**Handling Edge Cases:**
- Returns NaN where model = 0
- Propagates NaN from input arrays
- Works element-wise on arrays

### 3. Right-Edge Annotation

**Function:** `annotate_right()`

**Purpose:** Add labels to right edge of plot aligned with curves.

```python
from plotting import annotate_right

annotate_right(
    ax,
    ys_right=[5.2, 9.8, 13.5],      # Y-values at right edge
    labels=['263 K', '273 K', '283 K'],
    fontsize=12
)
```

**Features:**
- Automatic vertical spacing to prevent overlap
- Aligned to right axis
- Uses curve colors for matching
- Replaces traditional legend when many curves

### 4. Tight X-Limits

**Function:** `tight_xlim()`

**Purpose:** Set x-axis limits with controlled padding.

```python
from plotting import tight_xlim

tight_xlim(
    ax, x_data,
    pad_left=0.03,    # 3% padding on left
    pad_right=0.10    # 10% padding on right (for labels)
)
```

**Use Case:**
- Maximize data visibility
- Leave room for right-edge labels
- Consistent padding across plots

## Global Styling

### Default Settings

Located at top of `plotting.py`:

```python
# Global plot settings
AxLabelSize = 14      # Axis label font size
TLabelSize = 12       # Tick and legend font size
LineWidth = 2.0       # Line width for model curves
MarkerSize = 8        # Marker size for data points
```

### LaTeX Rendering

**Requirements:**
- Working TeX installation
- STIX fonts
- siunitx package
- upgreek package
- mhchem package

**Configuration:**
```python
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
```

### Figure Export

**High-resolution:**
```python
fig.savefig('plot.pdf', dpi=300, bbox_inches='tight')
fig.savefig('plot.png', dpi=300, bbox_inches='tight')
```

**For presentations:**
```python
fig.savefig('plot.png', dpi=150, bbox_inches='tight')
```

**For web:**
```python
fig.savefig('plot.png', dpi=72, bbox_inches='tight')
```

## Common Patterns

### Study Plotting Script

Template for analysis-specific plotting:

```python
#!/usr/bin/env python3
"""
Plotting script for [Study Name] conductivity data.
"""

import numpy as np
import pandas as pd
from plotting import (plot_sigma_vs_concentration,
                     plot_sigma_vs_temperature,
                     plot_conductivity_vs_pressure)

# Load curated data
data = pd.read_csv('data/study_name/curated_data.csv')

# Extract NaCl measurements
nacl = data[data['comp'] == 'NaCl']

# Group by concentration
concs = nacl['w_molal'].unique()

# Plot σ vs T for each concentration
temps = nacl['T_K'].unique()
for conc in concs:
    subset = nacl[nacl['w_molal'] == conc]

    fig = plot_sigma_vs_temperature(
        temp_data=subset['T_K'].values,
        sigma_data=[subset['conductivity_Sm'].values],
        conc_labels=[f'{conc:.1f} mol/kg'],
        sigma_errors=[subset['conductivity_Sm'].values *
                     subset['S_unc_pct'].values / 100],
        title=f'NaCl {conc:.1f} mol/kg',
        out_file=f'nacl_{conc:.1f}m_sigma_vs_T.pdf'
    )
```

### Combined Plot (σ vs m and σ vs T)

```python
# Create side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: σ vs m at fixed T
plot_sigma_vs_concentration(
    conc_data=concs,
    sigma_data=sigma_at_273K,
    temp_labels=['273 K'],
    title='(a) σ vs Concentration'
)

# Right: σ vs T at fixed m
plot_sigma_vs_temperature(
    temp_data=temps,
    sigma_data=sigma_at_1m,
    conc_labels=['1.0 mol/kg'],
    title='(b) σ vs Temperature'
)

plt.tight_layout()
fig.savefig('combined_plots.pdf', dpi=300)
```

## Integration with Analysis Workflow

### From GUI to Publication Plot

```bash
# 1. Run GUI analysis
python gamry_HiPOZ.py

# 2. Export curated data (GUI: "Export Plots to PDF")
# Creates: data/<date>/curated_data.csv

# 3. Generate publication plots
python plot_study.py

# 4. Generate LaTeX tables
python generate_cortes_latex_tables.py

# 5. Include in manuscript
# \input{tables/conductivity_data.tex}
# \includegraphics{figures/sigma_vs_conc.pdf}
```

### Automation

```bash
#!/bin/bash
# plot_all.sh - Generate all plots for study

python plot_study.py --compound NaCl
python plot_study.py --compound KCl
python plot_study.py --compound MgSO4

python generate_cortes_latex_tables.py
```

## See Also

- [PLOTTING.md](PLOTTING.md) - General plotting guide
- [GUI_OVERVIEW.md](GUI_OVERVIEW.md) - GUI documentation
- [CALIBRATION.md](CALIBRATION.md) - Analysis workflow
- [Mahboub README](../mahboub2026/README.md) - Example study plots
- [Cortes README](../cortes2026/README.md) - Another example
