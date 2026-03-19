# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HiPOZ (High-Pressure Ocean world analog impedance (Z)) is a Python application for measuring and analyzing electrical impedance spectroscopy data from high-pressure fluid experiments. The system processes impedance measurements from Gamry instruments, fits equivalent circuit models, and provides interactive data curation tools.

## Environment Setup

### Required Dependencies

Install via conda:
```bash
conda install numpy matplotlib
```

Install via conda-forge:
```bash
conda install -c conda-forge schemdraw
```

Install via pip:
```bash
pip install impedance SeaFreeze PlanetProfile bidict PyQt5 tqdm
```

**Important:** A working TeX installation is required for Matplotlib rendering. The plotting system uses LaTeX for figure labels and requires STIX fonts, siunitx, upgreek, and mhchem packages.

## Running the Application

### Main Entry Point

```bash
python gamry_HiP.py
```

This launches the PyQt5 GUI application that:
1. Loads impedance data from `data/` subdirectories
2. Fits equivalent circuit models to each measurement
3. Opens an interactive data selector window for analysis and curation

### Automated Calibration (New!)

You can now automate the calibration workflow using configuration files:

```bash
# Auto-detect config in data directory
python gamry_HiP.py

# Or specify config explicitly
python gamry_HiP.py --config calibration_config.json
```

Place a `calibration_config.json` or `calibration.json` file in your data directory to automatically:
- Mark calibration standards
- Calculate cell constant
- Compute conductivity for measurements
- Generate and save plots

See `CALIBRATION_README.md` for detailed instructions and examples.

Generate example configs:
```bash
python calibration_config.py
```

### Key Configuration

Edit the `dates` variable in `gamry_HiP.py` to specify which data directories to process:
```python
dates = ['RoseData']  # Process data from data/RoseData/
```

Circuit fitting type can be changed via:
```python
circType = 'CPE'  # Options: 'CPE', 'RC', 'RC-R', or custom circuit string
```

## Architecture

### Core Data Structures

**`gamryTools.py`** - Central library defining key classes:

- **`Solution`**: Represents a single impedance measurement
  - Loads data from Gamry instrument text files
  - Stores metadata (P, T, composition, frequency range)
  - Methods: `loadFile()`, `FitCircuit()`, `Recipe()`, `CalcConc()`
  - Fits equivalent circuit models to extract resistance

- **`ResistorData`**: Similar to Solution but for resistor calibration measurements

- **`CalStdFit`**: Manages conductivity standard calibration curves
  - Interpolates KCl standard bottle values as function of temperature
  - Used to compute cell constant from calibration measurements

- **`TimeSeries`**: Organizes multiple measurements chronologically
  - Methods: `organizeData()` - sorts and categorizes cal vs measurement data
  - Distinguishes KCl calibration standards from NaCl measurements

### Data Flow

1. **Data Loading** (`gamry_HiP.py`):
   - Scans `data/<date>/ConductivityData_Default/*.txt` for measurement files
   - Each file contains: timestamp, P, T, description, frequency sweep data
   - Creates Solution objects and loads impedance spectra

2. **Circuit Fitting** (`Solution.FitCircuit()`):
   - Fits equivalent circuit model (CPE: R₀ + parallel(R₁-CPE₁, C₁))
   - Uses `impedance.models.circuits.CustomCircuit`
   - Extracts DC resistance (R_calc) and uncertainty
   - Optional: Basin hopping optimization with multiprocessing

3. **Interactive Curation** (`DataSelector` GUI):
   - Timeseries plots of impedance vs time
   - Table view with editable P, T, composition, conductivity values
   - **Mark as Standard**: Select calibration rows → computes cell constant K_cell = σ × R
   - **Associate Measurements**: Apply K_cell to selected rows → compute σ = K_cell / R
   - Bode & Nyquist plots for selected measurements
   - S vs P scatter plot (conductivity vs pressure) colored by temperature
   - Auto-saves curated data to `hipoz_exports/` directory

4. **Visualization** (`gamryPlots.py`):
   - Nyquist plots (Re(Z) vs -Im(Z))
   - Bode plots (|Z| and phase vs frequency)
   - Time series plots with error bars
   - Conductivity vs P, T plots

### File Organization

```
data/
  <date>/
    ConductivityData_Default/
      Default_<timestamp>_P_<MPa>_T_<K>.txt  # Gamry impedance data files
    Default_PressTemps.txt                    # Pressure/temperature log
    Default_calibration.txt                   # Calibration metadata
```

**Data File Format** (Gamry output):
- Lines 1-9: Metadata (timestamp, T, P, description, drive voltage, frequency range)
- Line 10+: Data columns (index, frequency_Hz, |Z|_ohm, phase_deg)

### Important Implementation Details

**Equivalent Circuit Models:**
- **CPE** (default): `R₀-p(R₁-CPE₁, C₁)` - Captures electrode polarization
- **RC**: `p(R₁, C₁)` - Simple parallel RC (Pan et al. 2021 model)
- **RC-R**: `p(R₁, C₁)-R₀` - Adds series resistance
- Custom: Provide circuit string following `impedance` package syntax

**Circuit Fitting Notes:**
- Initial guess: `[R0_initial, R1_val, CPE_Q, CPE_n, C_val]`
- Uses L-BFGS-B bounded optimization
- Frequency filtering: `f_range_Hz = [10e3, 100e3]` typical range
- R₀ (first parameter) represents DC resistance for conductivity calculation

**Concentration Parsing:**
- Automatically extracts composition from file description field
- Supported solutes: 'DIwater', 'KCl', 'NaCl', 'MgSO4' (from PlanetProfile)
- Converts between ppt (g/kg solution) and molal (mol/kg solvent)
- KCl standards: {23, 84, 447, 2070, 2764, 15000, 80000} µS/cm

**Cell Constant Determination:**
- Select calibration measurements with known σ (from bottle label)
- Compute K_cell = σ_std × R_measured for each standard
- Average multiple standards for final K_cell value
- Uncertainty: combines measurement uncertainty with standard deviation

## Common Development Tasks

### Adding Support for New Circuit Models

1. Add circuit definition to `Solution.FitCircuit()` in `gamryTools.py`
2. Define circuit string using `impedance` package syntax
3. Provide appropriate initial guess array
4. Optionally add circuit diagram generation using `schemdraw`

### Processing New Data

1. Place Gamry output files in `data/<date>/ConductivityData_Default/`
2. Update `dates` list in `gamry_HiP.py`
3. Run: `python gamry_HiP.py`
4. Use GUI to mark calibration standards and associate measurements

### Modifying Data Selector UI

- Main window: `DataSelector.__init__()` in `DataSelector.py`
- Tab structure: Timeseries, Bode & Nyquist, S vs P
- Table editing: handled by `on_table_item_changed()`
- Button actions: `mark_as_standard()`, `associate_measurements()`, `create_plots()`

### Export and Data Persistence

- Curated data saved to: `hipoz_exports/hipoz_<timestamp>_curated.csv`
- Plots exported as PNG (300 dpi) and PDF
- Save modes: "overwrite" (default), "timestamp", "rolling" (keeps last N)
- Modify `save_curated_outputs()` to customize export behavior

## PlanetProfile Integration

The `configPP*.py` files provide integration with PlanetProfile for planetary science applications:
- `configPP.py` - General runtime parameters
- `configPPinduct.py` - Magnetic induction settings
- `configPPplots.py` - Plot configuration
- `configPPtrajec.py` - Spacecraft trajectory settings

These use PlanetProfile's thermodynamic models (SeaFreeze, MgSO4Props) for:
- Converting between concentration units
- Computing solution densities
- Calculating recipes for mixing standards

## Testing Approach

When modifying circuit fitting or data processing:
1. Use a small test dataset (single date folder)
2. Verify circuit fits converge (check log output)
3. Confirm GUI displays data correctly
4. Test calibration workflow: mark standard → associate → verify computed σ values
5. Check exported CSV contains expected columns and values
