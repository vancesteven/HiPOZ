# Cortes et al. (2026) - Conductivity Analysis

## Overview

This folder contains benchtop conductivity measurements and Gamry impedance data analysis for the Cortes et al. (2026) study.

## Data Sources

### Benchtop Conductivity Data
- **Raw data**: `../JesusData2025.csv` (needs cleaning)
- **Cleaned data**: `Cortes2026BenchtopData.csv` (to be created)
- **Instrument**: TBD (add details)
- **Temperature range**: 5°C, 10°C, 20°C
- **Compounds**: KCl, NaCl, MgSO4, Na2SO4, mixtures, organic acids (glycine, alanine, aspartic acid, glutamic acid)
- **Concentrations**: Various (0.5-2.0 M/L ranges)

### Gamry Impedance Data
- **Location**: `../JesusCortes/Data/`
- **Date folders**: 08.12.25, 08.13.25, 08.14.25, 08.15.25
- **Measurement types**:
  - Calibration standards (15 mS/cm, 80 mS/cm)
  - NaCl pressure series (0.5M, 0.75M, 1.0M, 1.5M, 2.0M)
  - NaCl + MgSO4 mixtures
  - Pressure range: 0-600 MPa
  - Temperature: Room temperature (~20°C)

## Files

- `cortes2026_plots.py` - Main plotting and analysis script
- `Cortes2026BenchtopData.csv` - Cleaned benchtop data (standardized format)
- `cortes_plots/` - Output directory for figures
- `archive/` - Archived versions and development files
- `tests/` - Unit tests for analysis functions

## Data Format

### Required CSV format for benchtop data:
```
compound,concentration_molal,temperature_C,temperature_K,conductivity_Sm,replicate,source,notes
```

Columns:
- `compound`: Chemical formula (e.g., KCl, NaCl, NaCl+MgSO4)
- `concentration_molal`: Molality (mol/kg_H2O)
- `temperature_C`: Temperature (Celsius)
- `temperature_K`: Temperature (Kelvin)
- `conductivity_Sm`: Conductivity (S/m)
- `replicate`: Replicate number (1, 2, 3, ...)
- `source`: Data source ('benchtop', 'Gamry', etc.)
- `notes`: Additional notes

## TODO

1. **Data cleaning**: Convert `../JesusData2025.csv` from spreadsheet format to standardized CSV format
2. **Concentration units**: Determine if concentrations are molal or molar, convert as needed
3. **Compound naming**: Standardize compound names and mixture notation
4. **Gamry data processing**: Wait for user instructions on organizing high-pressure data
5. **Analysis pipeline**: Update `cortes2026_plots.py` to work with cleaned data

## Usage

```bash
# Run analysis (once data is cleaned)
python cortes2026_plots.py
```

## Output filenames

Plot filenames are derived from compound names via `safe_filename_stem()` in
`cortes2026_plots.py`. Because compound/mixture names contain colons (e.g.
`Na2SO4:KCl_2:1`, `NaCl:MgSO4_1:1`) and colons are illegal in Windows
filenames, the helper sanitizes names before saving:

- spaces -> `_`
- colons (`:`) -> `-`

So `Na2SO4:KCl_2:1` produces `na2so4-kcl_2-1_vs_temperature.pdf` and
`NaCl:MgSO4_1:1` produces `nacl-mgso4_1-1_vs_concentration.pdf`. This keeps the
output cross-platform (macOS/Linux/Windows) compatible.

## Notes

- Original data file `JesusData2025.csv` is in Excel-style spreadsheet format
- Data includes interesting organic compound measurements (amino acids)
- Gamry data includes high-pressure measurements up to 600 MPa
- Multiple calibration standards used: 15 mS/cm and 80 mS/cm
