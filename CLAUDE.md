# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HiPOZ (High-Pressure Ocean world analog impedance (Z)) is a Python application for measuring and analyzing electrical impedance spectroscopy data from high-pressure fluid experiments. The system processes impedance measurements from Gamry instruments, fits equivalent circuit models, and provides interactive data curation tools.

## Environment Setup

### Required Dependencies

Dependencies are declared in `pyproject.toml` (pip-installable) and
`environment.yml` (conda-only pieces). Install in editable mode so edits to the
source take effect immediately:

```bash
# Full environment from scratch (recommended)
conda env create -f environment.yml
conda activate hipoz
pip install -e .

# Into an existing environment
pip install -e .

# With test dependencies
pip install -e ".[dev]"
```

Verified on CPython 3.11 and 3.14 (macOS arm64). Python 3.14 requires
numpy>=2.5, pandas>=3.0, matplotlib>=3.11 and PyQt6, which the version floors in
`pyproject.toml` allow.

**Reaktoro** is needed only for WATEQ4F speciation
(`compute_mccleskey_model(speciation=True)`, `speciation.py`) and for
PlanetProfile's CustomSolution ocean EOS. It is distributed on conda-forge only,
so `pip` cannot install it:

```bash
conda install -c conda-forge reaktoro
```

Everything else works without it; `speciation=True` raises a clear ImportError.

**Important:** A working TeX installation is required for Matplotlib rendering. The plotting system uses LaTeX for figure labels and requires STIX fonts, siunitx, upgreek, and mhchem packages. Install MacTeX/TeX Live separately — conda's `texlive-core` generally does not provide these packages.

## Running the Application

### Main Entry Point

```bash
python gamry_HiPOZ.py
```

This launches the PyQt5 GUI application that:
1. Loads impedance data from `data/` subdirectories
2. Fits equivalent circuit models to each measurement
3. Opens an interactive data selector window for analysis and curation

### Automated Calibration (New!)

You can now automate the calibration workflow using configuration files:

```bash
# Auto-detect config in data directory
python gamry_HiPOZ.py

# Or specify config explicitly
python gamry_HiPOZ.py --config calibration_config.json
```

Place a `calibration_config.json` or `calibration.json` file in your data directory to automatically:
- Mark calibration standards
- Calculate cell constant
- Compute conductivity for measurements
- Generate and save plots

See `docs/CALIBRATION.md` for detailed instructions and examples.

Generate example configs:
```bash
python analysis_config.py
```

### Key Configuration

Edit the `dates` variable in `gamry_HiPOZ.py` to specify which data directories to process:
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

- **`DataSelector`**: Interactive PyQt5 GUI for data curation (`hipoz_data_selector_gui.py`)
  - Four-tab interface: Data Table, Timeseries, Bode & Nyquist, S vs P
  - Editable table with columns: Filename, Calibration, Time, Comp, w(ppt), w(molal), T, P, Z, Z±, S, S±
  - Workflow buttons: Mark as Standard, Associate Measurements, Bulk Edit, Create Plots, Export
  - Auto-saves to config files (zAnalysis<date>.csv or .json)
  - Supports multi-component solutions with comma-separated values
  - Precision-matching for concentration conversions

### Data Flow

1. **Data Loading** (`gamry_HiPOZ.py`):
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
2. Update `dates` list in `gamry_HiPOZ.py`
3. Run: `python gamry_HiPOZ.py`
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

---

## Agent lanes

Claude Code is the **manager lane**: planning, scientific review and
adjudication, integration, and pushes. Delegate implementation and
reconnaissance passes to subagents; keep adjudication for the manager.

Codex is the **delegate lane**, restricted to tasks explicitly queued in
`plans/CODEX-QUEUE.md`. Its instructions are `AGENTS.md`. It commits locally and
never pushes. **Scientific adjudication is never delegated to Codex.**

Codex is mechanically review-only in the sandbox (`codex exec --sandbox
read-only`). Request a review with `codex-review <repo-dir> [git-range]`; the
wrapper writes `coordination/reviews/<timestamp>-<sha>.md` and appends its own
audit entry.

## State files

Forward-looking — what to do next, and who owns it:

- `plans/STATUS.md` — current focus, work in flight, blockers
- `plans/CODEX-QUEUE.md` — Codex's queue and the claim/report protocol

Backward-looking — what happened, and cross-repo messages:

- `coordination/audit.md` — append-only, one entry per completed action
- `coordination/inbox/<agent>.md` — messages addressed to an agent in this repo
- `coordination/open-questions.md` — anything needing Steve's sign-off
- `~/src/coordination/inbox/<agent>.md` — **cross-repo** messages (e.g. thrak ↔ lov3d)

**Queue freshness:** any session that pushes commits, integrates artifacts, or
changes a queue must refresh the `Updated:` lines and affected sections of
`plans/STATUS.md` and the relevant queue file **in the same session**.

Sibling repos are all visible under `~/src`. Never edit another repo's files
directly — hand work across via its inbox.

## Verification discipline

A change is NOT "done" until its specified behavior has been observed in the
running system. Compiling, importing, or running a smoke pass that does not
exercise the specific change is not verification.

For this repo: a targeted `pytest` test under `tests/` must exercise the changed path. For any UI or plotting change, the app must be launched, the interaction reproduced, and the behavior visually confirmed — `py_compile` passing is not verification.

If you cannot verify (no env, no display, no time), say so explicitly and label
the change `implemented, unverified` — never `done` or `fixed`.

## Status vocabulary

Status entries must use exactly this vocabulary:

- `verified` — observed working under documented reproduction steps; **cite the
  artifact** (test name, PDF path, screenshot).
- `implemented, unverified` — code written and syntax-checked, but the targeted
  behavior has not been observed.
- `not implemented` — planned, no code written.

Do NOT use `done`, `fixed`, `complete`, `syntax clean`, or `review passed`.
Those describe intermediate states, not whether the problem is gone.

**Layering a new fix on top of an `implemented, unverified` change is forbidden
without verifying the prior change first.** A chain of unverified fixes is a
chain of unknowns.

## Escalation

Stop and write to `coordination/open-questions.md` — do not proceed — for
anything ambiguous, destructive (deletions, force-pushes, dependency upgrades),
or scientifically consequential. **Never change a scientific assumption to make
a test or gate pass.**

Commit coordination and plan files with the work they describe, so the record
and the code state cannot drift apart.
