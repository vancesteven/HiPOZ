# Mahboub et al. (2026, in press) Data Integration

This document explains how the Mahboub et al. (2026) benchtop conductivity data is integrated with HiPOZ and Gamry impedance measurements.

## Overview

The Mahboub et al. (2026, in press) study measured conductivity of various salt solutions (NaCl, MgSO₄, NH₄Cl, Na₂CO₃, and mixtures) using benchtop probes across a range of temperatures (-10°C to 25°C) and concentrations.

This data is now:
1. **Stored in CSV format** (`Mahboub2026BenchtopData.csv`) for easy editing
2. **Integrated with Gamry data** from HiPOZ impedance measurements
3. **Plottable using general functions** from `plotting.py`

## File Structure

### Data Files

**`Mahboub2026BenchtopData.csv`** - All benchtop probe measurements
- Format: CSV with columns for compound, concentration, temperature, conductivity, replicate, source
- Contains:
  - NaCl: 7 temperatures × 6 concentrations × 3 replicates = 126 measurements
  - MgSO4: 7 temperatures × 6 concentrations × 3 replicates = 126 measurements
  - NH4Cl: 7 temperatures × 4 concentrations × 3 replicates = 84 measurements
  - Na2CO3: 3 temperatures × 5 concentrations × 3 replicates = 45 measurements
  - Mixture: 4 temperatures × 4 concentrations × 3 replicates = 48 measurements
  - McCleskey 2011 reference data
  - Gamry measurements from literature

### Code Modules

**`plotting.py`** - General plotting functions (dataset-agnostic)
- `plot_sigma_vs_concentration()` - Plot σ vs concentration with multiple temperature curves
- `plot_sigma_vs_temperature()` - Plot σ vs temperature with multiple concentration curves
- `plot_nyquist()`, `plot_bode()` - Gamry impedance plots
- `group_data_by_variable()` - General data organization function
- `safe_errorbar()`, helper functions

**`mahboub2026_data.py`** - Mahboub-specific data loading
- `load_mahboub_benchtop_data()` - Load CSV data
- `organize_by_temperature()` - Organize for σ vs conc plots
- `organize_by_concentration()` - Organize for σ vs temp plots
- `load_gamry_from_hipoz()` - Load Gamry measurements

**`mahboub2026_plots.py`** - Script to reproduce all Mahboub plots
- Generates 10 plots (5 compounds × 2 plot types)
- Overlays Gamry data where available
- Outputs to `mahboub_plots/` directory

## Usage

### Quick Start

```bash
# Reproduce all Mahboub plots
python mahboub2026_plots.py
```

This generates:
- `mahboub_plots/nacl_vs_concentration.pdf`
- `mahboub_plots/nacl_vs_temperature.pdf`
- `mahboub_plots/mgso4_vs_concentration.pdf`
- ... (10 plots total)

### Custom Plotting

```python
from mahboub_data import load_mahboub_benchtop_data, organize_by_temperature
from plotting import plot_sigma_vs_concentration

# Load data
data = load_mahboub_benchtop_data()

# Organize NaCl data by temperature
conc, sigma_by_temp, errors, temp_labels = organize_by_temperature(data, 'NaCl')

# Plot
plot_sigma_vs_concentration(
    conc_data=conc,
    sigma_data=sigma_by_temp,
    sigma_errors=errors,
    temp_labels=temp_labels,
    title='NaCl Conductivity',
    out_file='my_nacl_plot.pdf'
)
```

### Overlay Gamry Data

```python
from mahboub_data import load_gamry_from_hipoz

# Load Gamry data
gamry_data = load_gamry_from_hipoz(
    'data/20250813Mahboub2026/hipoz_latest_curated.csv',
    compound='NaCl'
)

# Plot benchtop data
fig = plot_sigma_vs_concentration(...)

# Overlay Gamry points
ax = fig.axes[0]
ax.errorbar(gamry_data['concentration_molal'],
           gamry_data['conductivities_Sm'],
           yerr=gamry_data['errors'],
           fmt='o', color='red', label='Gamry')
ax.legend()
fig.savefig('nacl_comparison.pdf')
```

## Data Format Details

### CSV Structure

```csv
compound,concentration_molal,temperature_C,temperature_K,conductivity_Sm,replicate,source,notes
NaCl,0.1711,25,298.15,1.788,1,benchtop,10 g/kg / 58.44
NaCl,0.1711,25,298.15,1.777,2,benchtop,
NaCl,0.1711,25,298.15,1.753,3,benchtop,
...
```

### Source Categories

- **benchtop** - Mahboub benchtop probe measurements
- **Gamry20250813** - HiPOZ Gamry impedance from 20250813
- **GamryLiterature** - Published Gamry measurements
- **McCleskey2011** - McCleskey et al. 2011 reference data

### Units

- **Concentration**: mol/kg (molality)
- **Temperature**: Both °C and K provided
- **Conductivity**: S/m (convert µS/cm by dividing by 10,000)

## Creating Similar Datasets

To create a similar dataset for future studies:

### 1. Create CSV File

Follow the same format as `Mahboub2026BenchtopData.csv`:

```csv
compound,concentration_molal,temperature_C,temperature_K,conductivity_Sm,replicate,source,notes
YourSalt,0.5,20,293.15,5.5,1,benchtop,Your notes here
YourSalt,0.5,20,293.15,5.6,2,benchtop,
...
```

### 2. Create Data Loading Module

Create `your_study_data.py`:

```python
import pandas as pd
import numpy as np

def load_your_study_data(file_path='YourStudy2026Data.csv'):
    """Load data from your CSV file."""
    df = pd.read_csv(file_path, comment='#')

    data = {}
    for compound in df['compound'].unique():
        comp_data = df[df['compound'] == compound].copy()
        data[compound] = {
            'concentration_molal': comp_data['concentration_molal'].values,
            'temperatures_K': comp_data['temperature_K'].values,
            'conductivities_Sm': comp_data['conductivity_Sm'].values,
            # ... other fields
        }

    return data

def organize_by_temperature(data, compound):
    """Organize for σ vs concentration plots."""
    # Implementation similar to mahboub2026_data.py
    # ...
    return conc_array, sigma_by_temp, errors, temp_labels
```

### 3. Use General Plotting Functions

```python
from plotting import plot_sigma_vs_concentration
from your_study_data import load_your_study_data, organize_by_temperature

data = load_your_study_data()
conc, sigma, errors, labels = organize_by_temperature(data, 'YourSalt')

plot_sigma_vs_concentration(conc, sigma, errors, labels, out_file='plot.pdf')
```

## Integration with Gamry Data

### Workflow

1. **Measure with Gamry**: Use HiPOZ to process impedance data
   ```bash
   python gamry_HiPOZOZ.py --dates 20250815
   ```

2. **Mark standards** in GUI → exports `hipoz_latest_curated.csv`

3. **Load both datasets**:
   ```python
   benchtop = load_mahboub_benchtop_data()
   gamry = load_gamry_from_hipoz('hipoz_latest_curated.csv')
   ```

4. **Plot comparison** using `plot_sigma_vs_concentration()`

### Data Comparison

| Method | Precision | Time | Cost | Notes |
|--------|-----------|------|------|-------|
| Benchtop probe | ±3% | Seconds | Low | Quick, direct |
| Gamry impedance | ±4-7% | Minutes | Medium | Frequency-dependent info |
| Literature | Varies | N/A | Free | Reference validation |

## Error Propagation

### Mahboub Benchtop Data

Total error = √(σ_random² + σ_systematic²)

Where systematic components:
- Probe calibration: 0.5%
- Temperature control: 2.89%
- Concentration preparation: 0.5%

### Gamry Data

Total error from HiPOZ:
- Circuit fit uncertainty
- Cell constant uncertainty
- Combined using standard error propagation

## Comparison with Original MahboubEtAl2026.py

### Original Script
- **Data**: Hard-coded numpy arrays
- **Plots**: Custom plotting for each compound
- **Gamry integration**: Manual array manipulation
- **Reusability**: Limited

### New Modular System
- **Data**: CSV file (Excel-editable)
- **Plots**: General functions from `plotting.py`
- **Gamry integration**: Automatic via `mahboub2026_data.py`
- **Reusability**: High - template for future studies

### Advantages

1. **Easy updates**: Edit CSV, no code changes needed
2. **Version control**: Track data changes separately from code
3. **Extensible**: Add new compounds or sources easily
4. **Reproducible**: Single command reproduces all plots
5. **Documented**: Clear data provenance in CSV

## References

- Mahboub, R., et al. (2026). [In preparation]
- McCleskey, R.B., et al. (2011). Electrical conductivity of electrolytes...
- HiPOZ documentation: CLAUDE.md, README.md

## Contact

For questions about Mahboub data integration:
- Steven D. Vance (steven.d.vance@jpl.nasa.gov)
