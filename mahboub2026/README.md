# Mahboub et al. (2026) Conductivity Analysis

This directory contains the complete analysis and plots for the Mahboub et al. (2026) publication on benchtop conductivity measurements of salt solutions.

## Citation

Mahboub, R., et al. (2026). [Paper title]. *Journal Name*.

## Contents

- **`mahboub2026_plots.py`** - Main plotting script (generates all 10 figures)
- **`Mahboub2026BenchtopData.csv`** - All benchtop and Gamry impedance data
- **`mahboub_plots/`** - Directory containing 10 publication-quality PDF plots
- **`MAHBOUB_CHANGES.md`** - Detailed changelog of analysis development

## Quick Start

### Regenerate All Plots

From the `mahboub2026/` directory:

```bash
python mahboub2026_plots.py
```

This will regenerate all 10 plots in the `mahboub_plots/` directory:
- NaCl: concentration and temperature plots
- MgSO₄: concentration and temperature plots
- NH₄Cl: concentration and temperature plots
- Na₂CO₃: concentration and temperature plots
- Mixture (1:1:1): concentration and temperature plots

### Customize Plot Settings

Edit `../config_plots.py` to change:
- Font sizes (labels, title, legend)
- Colormaps (tab10, plasma, viridis, etc.)
- Plot options (show legend, show title, show delta subplots)
- Output resolution (DPI)

Changes will apply to all plots when regenerated.

## Data Format

### Benchtop Data: `Mahboub2026BenchtopData.csv`

This CSV contains **only** benchtop conductivity measurements from the Thermo Scientific ORION Star A329 conductivity meter. Gamry impedance data are analyzed separately from raw Gamry files (see `../data/20250813Mahboub2026/`).

**Columns:**
- `compound` - Chemical formula (NaCl, MgSO4, NH4Cl, Na2CO3, Mixture)
- `concentration_molal` - Molality (mol/kg_H2O)
- `temperature_C` - Temperature in Celsius
- `temperature_K` - Temperature in Kelvin
- `conductivity_Sm` - Conductivity in S/m
- `replicate` - Replicate number (1, 2, 3)
- `source` - Data source (all entries are `benchtop`)
- `notes` - Additional notes (e.g., original concentration in g/kg)

**Total measurements:**
- 429 benchtop probe measurements (3 replicates per condition)
- All measurements at 7 temperatures: -10°C, -6°C, -3°C, -1°C, 5°C, 20°C, 25°C

### Gamry Impedance Data

Gamry impedance measurements are **not** included in the CSV. They are analyzed separately through the HiPOZ impedance analysis pipeline from raw Gamry files located in:
- `../data/20250813Mahboub2026/` - Initial MgSO₄ measurements
- `../data/20250815Mahboub2026/` - Additional MgSO₄ measurements

Run the impedance analysis first:
```bash
python gamry_HiPOZ.py --dates 20250813Mahboub2026 20250815Mahboub2026
```

Results are saved to `hipoz_*_results.csv` in each data directory and automatically combined by the plotting script.

Gamry data appear as overlays on concentration plots but are processed independently from raw impedance spectra.

## Plot Features

### All Plots Include:
- ✅ Publication-quality LaTeX formatting
- ✅ Tab10 discrete colormap (matches paper)
- ✅ Error bars (systematic + random uncertainty)
- ✅ 300 DPI resolution

### Concentration Plots Feature:
- σ vs concentration curves (one per temperature)
- Gamry impedance overlays (NaCl and MgSO₄)
- Delta (Δ%) deviation subplots from McCleskey model
- Automatic frozen-sample filtering (MgSO₄ at -10°C)

### Temperature Plots Feature:
- σ vs temperature curves (one per concentration)
- Delta (Δ%) deviation subplots from McCleskey model
- Right-edge concentration labels
- Automatic frozen-sample filtering

## Key Analysis Features

### 1. Frozen Sample Filtering
MgSO₄ at -10°C (263.15 K) and lowest concentration (0.0249 mol/kg) shows zero conductivity due to freezing. These points are automatically filtered (set to NaN) and not displayed.

**Implementation:** `filter_frozen=True` in organization functions detects conductivity < 1e-12 S/m.

### 2. McCleskey Model Comparison
All single-salt plots include Delta (Δ%) subplots showing deviation from McCleskey et al. (2012) empirical model:

Δ% = 100 × (σ_measured - σ_model) / σ_model

**Note:** Mixture plots exclude Delta subplots (no model for ion mixtures).

### 3. Gamry Impedance Overlays
- **MgSO₄:** 4 concentrations from HiPOZ impedance analysis (data/20250815Mahboub2026)

Gamry data shown with black outlines and temperature-matched color fill. Other compounds may have Gamry data added as additional measurements are analyzed.

### 4. Error Propagation
Total uncertainty combines:
- Random error (standard deviation of replicates)
- Probe calibration: 0.5%
- Temperature control: 2.89%
- Concentration preparation: 0.5%

σ_total = √(σ_random² + (0.005×σ)² + (0.0289×σ)² + (0.005×σ)²)

## Dependencies

Required Python packages:
```bash
pip install numpy pandas matplotlib
```

Required modules (in parent directory):
- `plotting.py` - Low-level plotting functions
- `study_plots.py` - High-level study functions
- `config_plots.py` - Universal configuration
- `sigmaElectricMcCleskey2012.py` - McCleskey model

## Instrument Information

**Benchtop Probe:** Thermo Scientific ORION Star A329 Conductimeter
- Temperature range: -10°C to 25°C
- Precision: 0.5% (calibration) + 2.89% (temperature control)

**Gamry Impedance:** HiPOZ analysis pipeline
- Equivalent circuit fitting (CPE model)
- Cell constant determination from KCl standards
- Typical uncertainty: 4-7%

## File Sizes

```
Mahboub2026BenchtopData.csv:     ~50 KB
mahboub_plots/ (10 PDFs):        ~6.5 MB total
mahboub2026_plots.py:            ~15 KB
```

## Analysis Details

### Frozen Sample Handling
MgSO₄ at -10°C (263.15 K) and lowest concentration (0.0249 mol/kg) shows zero conductivity due to freezing. The analysis automatically:
- Detects conductivity < 1e-12 S/m
- Sets values to NaN to exclude from plots
- Prevents plotting artifacts and incorrect model comparisons

### Colormap Choices
- **Concentration plots:** Tab10 discrete palette (7 distinct colors for 7 temperatures)
- **Temperature plots:** Tab10 discrete palette (distinct colors for each concentration)
- Matches colors from published figures in Mahboub2026rev2.pdf

### Gamry Integration Workflow
1. Analyze impedance data: `python gamry_HiPOZ.py --dates 20250813Mahboub2026 20250815Mahboub2026`
2. Results saved to `hipoz_*_results.csv` in each directory with conductivity values
3. Plotting script automatically loads and combines data from both datasets
4. Error handling provides clear feedback if standards not properly defined

## Troubleshooting

### Missing or incomplete Gamry conductivity data

If you see warnings about missing conductivity data:

**"No conductivity data in Gamry results"**
- Cause: Standards not defined in calibration config
- Fix: Edit the appropriate config file:
  - `data/20250813Mahboub2026/zAnalysis20250813Mahboub2026.csv`
  - `data/20250815Mahboub2026/zAnalysis20250815Mahboub2026.csv`
- Add standards with known conductivity values
- Re-run: `python gamry_HiPOZ.py --dates 20250813Mahboub2026 20250815Mahboub2026`

**"All Gamry conductivity values are NaN"**
- Cause: Standards defined but not associated with measurements
- Fix: In HiPOZ GUI:
  1. Mark calibration rows as "Standard"
  2. Select measurement rows
  3. Click "Associate Measurements"
- Or manually edit the config file to assign measurements to groups
- Re-run impedance analysis

**"X/Y measurements missing conductivity"**
- Cause: Some measurements not associated with standards
- Impact: Only measurements with conductivity will appear in plots
- Fix: Associate remaining measurements with appropriate standards

The plotting script will skip Gamry overlays if conductivity data is unavailable, but will still generate all benchtop plots successfully.

### Missing LaTeX fonts
If plots fail with LaTeX errors, ensure you have:
- Working TeX installation (MacTeX, TeX Live, or MiKTeX)
- STIX fonts
- siunitx, upgreek, mhchem packages

### McCleskey model errors
If Delta subplots fail for custom compounds, add ion specification to `../study_plots.py`:
```python
ION_SPECS['MyCompound'] = {'Ion1_p1': 1.0, 'Ion2_m1': 1.0}
```

### Colors don't match paper
Verify `../config_plots.py` has:
```python
COLORMAP_CONCENTRATION = 'tab10'
COLORMAP_TEMPERATURE = 'tab10'
```

## Version History

**2026-03-19:** Refactored with generalized functions
- Extracted reusable Gamry integration to `gamry_integration.py`
- Reduced plotting script from 420 to 340 lines
- Comprehensive error handling and testing
- All 10 plots verified against paper

**2026-03-19:** Analysis features complete
- Frozen sample filtering (MgSO₄ at -10°C)
- Tab10 discrete colormap matching published figures
- Gamry impedance overlays (MgSO₄: 4 points)
- Delta (Δ%) deviation subplots for single salts
- Complete test suite (4/4 tests passing)

## Related Documentation

- `../STUDY_PLOTS_README.md` - Guide to generalized plotting system
- `../READINESS_REPORT.md` - Complete verification report
- `../config_plots.py` - Universal plot configuration

## Support

For questions about the plotting system or adapting for other studies, see the parent directory's documentation.
