# HiPOZ

**H**igh-**P**ressure **O**cean world analog impedance (**Z**) analyzer

Software for measuring and analyzing electrical impedance spectroscopy data from high-pressure fluid experiments.

---

> **Note:** This is the first major deployment of HiPOZ developed with assistance from Claude (Anthropic). While the code has been tested and verified, some documentation and code comments may benefit from rewording for clarity.
>
> **For users:** Work in the `main` branch for most applications. Report issues, suggestions, or unclear documentation at [github.com/vancesteven/hipozgenai/issues](https://github.com/vancesteven/hipozgenai/issues).
>
> **For contributors using GenAI:** If you'd like to contribute using your own GenAI tools, please work from your own branch in a worktree from `main`. This keeps AI-assisted development organized and allows for proper review before merging.

---

## Quick Start

### For Students (No Python Editing Required!)

```bash
# Run with GUI - will prompt for directory selection:
python gamry_HiPOZ.py

# Or specify directory from command line:
python gamry_HiPOZ.py --dates 20250815Mahboub2026
```

That's it! Select your data directory and the GUI will guide you through calibration.

### For Automated Analysis

```bash
# Headless mode (requires config file with standards):
python gamry_HiPOZ.py --headless --dates 20250815Mahboub2026

# Harmonize CSV and JSON after editing:
python gamry_HiPOZ.py --harmonize data/20250815/zAnalysis20250815.csv
```

## Features

### 🎯 New in This Version

- **No Python editing required** - Use GUI dialog or `--dates` flag instead of editing code
- **CSV/JSON config harmonization** - Edit in Excel, auto-sync to JSON
- **Headless batch processing** - Automated analysis without GUI
- **Computed conductivity write-back** - Results saved to original config files
- **Directory selection dialog** - Point-and-click data folder selection
- **Exclude feature** - Mark bad measurements to skip

### Core Capabilities

- Load impedance data from Gamry instruments
- Fit equivalent circuit models (CPE, RC, RC-R)
- Interactive data curation with PyQt6 GUI
- Calibration with KCl standards
- Cell constant calculation and conductivity determination
- Export to CSV with uncertainty quantification
- Generate publication-quality plots (Nyquist, Bode, S vs P)

## Installation

### Quick Install

Dependencies are declared in `pyproject.toml` (everything pip can install) and
`environment.yml` (the conda-only pieces). Install in editable mode so your edits
to the source take effect immediately:

```bash
# Full environment from scratch (recommended)
conda env create -f environment.yml
conda activate hipoz
pip install -e .

# Or into an existing environment
pip install -e .

# With test dependencies
pip install -e ".[dev]"
```

### Version Requirements

**Python:** 3.11 or newer. Verified on **3.11** and **3.14** (macOS arm64);
3.12 and 3.13 should work but have not been exercised.

Python 3.14 requires the newest release of each compiled dependency, because
those are the only versions with cp314 wheels:

| Package | 3.11 (tested) | 3.14 (required) |
|---|---|---|
| numpy | 1.26.4 | 2.5.2 |
| pandas | 2.3.1 | 3.0.5 |
| matplotlib | 3.10.5 | 3.11.1 |
| scipy | 1.16.1 | 1.18.0 |
| PyQt6 | 6.11.0 | 6.11.0 |

The floors in `pyproject.toml` accommodate both. Circuit fitting was checked to
be numerically equivalent across versions (identical data hashes; CPE fits agree
to ~1e-9, i.e. optimizer float noise).

**Note on numpy 2.x:** earlier versions of this README pinned numpy 1.26.4 out of
concern for the `impedance` package. That pin is no longer needed — `impedance`
1.7.1 works with numpy 2.5.2, and its upstream CI tests through Python 3.14.

**Note on Qt:** the GUI uses **PyQt6**. PyQt5 has had no release since July 2024
and is not actively validated against new Python versions. PyQt6 requires Python
3.10+, so it runs on 3.11 as well.

### Optional: Reaktoro (for speciation)

Needed only for WATEQ4F speciation of the McCleskey conductivity model
(`speciation.py`) and for PlanetProfile's CustomSolution ocean EOS. Reaktoro is
distributed on **conda-forge only** — `pip` cannot install it:

```bash
conda install -c conda-forge reaktoro
```

Everything else works without it; requesting `speciation=True` without Reaktoro
raises a clear ImportError.

### First-run note

Run HiPOZ from the repository directory. PlanetProfile looks for `configPP*.py`
in the current working directory and, if they are absent, prompts interactively
to copy its defaults:

```
configPP files not found in pwd: /some/other/dir. Copy from defaults to local dir? [y]/n
```

In a non-interactive context (a script, cron job, or piped command) that prompt
raises `EOFError`. Either `cd` to the repo first, or answer the prompt once in
the directory you intend to work from.

**Important:** A working TeX installation is required for Matplotlib rendering. The plotting system uses LaTeX for figure labels and requires STIX fonts, siunitx, upgreek, and mhchem packages. Install MacTeX/TeX Live separately — conda's `texlive-core` generally does not provide these packages.

## Usage

### 1. Organize Your Data

Place Gamry measurement files in this structure:
```
data/
  <date>/
    ConductivityData_Default/
      Default_<timestamp>_P_<MPa>_T_<K>.txt
```

### 2. Run Analysis

**Option A: GUI with Directory Selection**
```bash
python gamry_HiPOZ.py
```
- Dialog opens to select data directory
- GUI loads all measurements
- Follow on-screen instructions to mark standards and associate measurements

**Option B: Command Line**
```bash
python gamry_HiPOZ.py --dates 20250815Mahboub2026
```

**Option C: Headless Mode (Automated)**
```bash
# First run: Create config with GUI
python gamry_HiPOZ.py --dates 20250815

# Edit CSV in Excel to mark standards and add conductivity values

# Subsequent runs: Automated analysis
python gamry_HiPOZ.py --headless --dates 20250815
```

### 3. Edit Config File (Excel-Friendly!)

The GUI auto-creates a config file: `data/<date>/zAnalysis<date>.csv`

Open in Excel and fill in:
- **type**: `standard` or `measurement`
- **conductivity_Sm**: Known conductivity for standards (e.g., 0.0084 for 84 µS/cm KCl)
- **comp**: Composition (e.g., KCl, NaCl, MgSO4)
- **w_ppt**: Concentration in g/kg solution (parts per thousand)
- **w_molal**: Concentration in mol/kg solvent (molality)
- **exclude**: Mark as `x` to skip bad files
- **notes**: Any observations about the measurement

### 4. Harmonize Formats

After editing CSV in Excel:
```bash
python gamry_HiPOZ.py --harmonize data/20250815/zAnalysis20250815.csv
```
This creates/updates the matching JSON file.

## Configuration Files

### Naming Convention
- `data/<date>/zAnalysis<date>.csv` - Excel-friendly (recommended)
- `data/<date>/zAnalysis<date>.json` - Programmatic access

### Format
**CSV columns:**
```csv
group_name,filename,P_MPa,T_K,type,conductivity_Sm,comp,w_ppt,w_molal,exclude,notes
```

**JSON structure:**
```json
{
  "calibration_groups": [
    {
      "name": "Group 1",
      "standards": [
        {
          "filename": "Default_..._KCl.txt",
          "conductivity_Sm": 0.0084,
          "comp": "KCl",
          "w_ppt": 1000
        }
      ],
      "measurements": [
        {
          "filename": "Default_..._NaCl.txt",
          "comp": "NaCl",
          "w_ppt": 1000
        }
      ]
    }
  ]
}
```

## Output Files

All output saved to same directory as config file:

- `hipoz_<timestamp>_results.csv` - Detailed analysis results with uncertainties
- `zAnalysis<date>_analyzed.csv` - Updated config with computed conductivities
- `zAnalysis<date>_analyzed.json` - JSON version of updated config
- Plots (optional): Bode, Nyquist, S vs P

## Publication Plotting System

### Benchtop Conductivity Studies

HiPOZ includes a generalized plotting system for creating publication-quality conductivity plots from benchtop and Gamry measurements. Designed for reproducibility and consistency across multiple studies.

**Complete Studies:**
- **`mahboub2026/`** - Mahboub et al. (2026) analysis (10 plots, 5 compounds)

**Features:**
- ✅ σ vs concentration plots with temperature curves
- ✅ σ vs temperature plots with concentration curves
- ✅ σ vs pressure plots (for high-pressure studies)
- ✅ Automatic McCleskey model comparison (Δ% deviation subplots)
- ✅ Gamry impedance data overlays
- ✅ Publication-quality LaTeX formatting
- ✅ Universal configuration via `config_plots.py`
- ✅ CSV-based data storage (Excel-editable)

### Creating New Study Plots

**Quick Start:**
```bash
# 1. Create data CSV (see CortesDataTemplate.csv)
cp CortesDataTemplate.csv MyStudyData.csv
# Edit MyStudyData.csv with your measurements

# 2. Copy template
cp cortes_plots_template.py mystudy_plots.py
# Update compound names and settings

# 3. Configure appearance (optional)
# Edit config_plots.py to change fonts, colors, etc.

# 4. Generate plots
python mystudy_plots.py
```

**Documentation:**
- `STUDY_PLOTS_README.md` - Complete guide to plotting system
- `mahboub2026/README.md` - Example study documentation
- `config_plots.py` - Universal plot configuration
- `cortes_plots_template.py` - Template for new studies
- `CortesDataTemplate.csv` - Template for data format

**Modules:**
- `plotting.py` - Low-level plotting functions (general purpose)
- `study_plots.py` - High-level study functions (reusable)
- `config_plots.py` - Universal settings (fonts, colors, options)
- `[study]_plots.py` - Study-specific scripts

### McCleskey (2012) Model and WATEQ4F Speciation

McCleskey et al. (2012) — "MC12" — computes conductivity as a sum over **free,
charged** species, `sigma = SUM lambda_i(I,T) * m_i`, where the free molalities
come from a WATEQ4F speciation calculation. Neutral complexes such as
`MgSO4(aq)` are absent from the sum because they carry no current. That omission
*is* the speciation correction.

By default `compute_mccleskey_model()` uses total (analytical) molality, i.e. it
assumes full dissociation. Pass `speciation=True` to run the speciation step:

```python
from study_plots import compute_mccleskey_model

# Default: total molality, full dissociation assumed
sigma = compute_mccleskey_model(concs, temps_K, compound='MgSO4')

# WATEQ4F speciation (needs Reaktoro)
sigma = compute_mccleskey_model(concs, temps_K, compound='MgSO4',
                                speciation=True)
```

`plot_study_concentration` and `plot_study_temperature` accept the same keyword,
and `plotting.py` can draw both curves together (`model_data` dashed as `MC12`,
`model_data_corrected` dotted as `MC12 + WATEQ4F`).

**Why it matters.** Association is negligible for 1:1 salts but dominant for
2:2 ones. Against the Mahboub dataset, all six MgSO4 concentrations lie above
McCleskey's stated 0.01245 mol/kg validity limit (the lowest by 2x, the highest
by 133x), and speciation cuts the RMS error 8-fold:

| compound | MC12 (total) | MC12 + WATEQ4F |
|---|---|---|
| MgSO4 | 99.9% | **12.5%** |
| NaCl | 9.1% | 9.1% (unchanged) |
| NH4Cl | 4.8% | 4.8% (unchanged) |

NaCl, KCl and NH4Cl are returned fully dissociated by WATEQ4F, so speciation
leaves them alone — the conductivity-standard calibration path is unaffected.

Check any dataset with:

```bash
python validate_speciation.py                  # Mahboub data
python validate_speciation.py --dataset both   # plus Cortes
python validate_speciation.py --no-speciation  # no Reaktoro needed
```

> **Known limitation.** The `lambda_0` values in the parameter table for the
> charged complexes `NaSO4-` (357), `KSO4-` (234) and `NaCO3-` (188) are
> implausibly large next to `Cl-` (77 S cm2/mol). Since speciation is what first
> activates these species, they dominate wherever they form — `NaCO3-` alone
> supplies 51% of speciated Na2CO3 conductivity — and push sigma *upward*, which
> is backwards for an association correction. An equivalent-vs-molal convention
> error was ruled out. **Only `MgSO4`, `NaCl`, `KCl` and `NH4Cl` speciation
> should be trusted** (see `speciation.VERIFIED_COMPOUNDS`) until these are
> checked against MC12's published table. `validate_speciation.py` flags this.

For carbonate systems the input specification fixes the pH, which then dominates
speciation (0.3 mol/kg Na2CO3: pH 11.4 supplying Na+/CO3-2 versus pH 7.7
supplying Na+/HCO3-). `speciation.SALT_RECIPES` specifies the alkaline case —
what you get dissolving solid Na2CO3, closed to atmospheric CO2 — explicitly
rather than relying on a library default.

**Coefficient corrections.** The parameter table was updated to match
PlanetProfile commit `b6a34e2`, restoring dropped negative signs for `H_p1`,
`CO3_m2`, `HCO3_m1`, `Fe_p3` and `KSO4_m1`, plus a missing zero in `CO3_m2`
(`lam2: -0.00326` → `-0.000326`). Of these only `CO3_m2` affects the compounds
studied here; it shifts Na2CO3 predictions by about -0.8% at 20 °C and -2.2% at
40 °C.

## Common Workflows

### Workflow 1: First Time Analysis
```bash
# 1. Run GUI
python gamry_HiPOZ.py --dates 20250815

# 2. GUI creates zAnalysis20250815.csv

# 3. Mark standards, associate measurements, export

# 4. Config saved with your work - reproducible!
```

### Workflow 2: Add Notes in Excel
```bash
# 1. Open data/20250815/zAnalysis20250815.csv in Excel

# 2. Add notes, compositions, concentrations

# 3. Save and close Excel

# 4. Harmonize
python gamry_HiPOZ.py --harmonize data/20250815/zAnalysis20250815.csv

# 5. Re-run analysis with updated metadata
python gamry_HiPOZ.py --dates 20250815
```

### Workflow 3: Batch Processing
```bash
# Process multiple days at once
python gamry_HiPOZ.py --dates 20250813 20250814 20250815

# Or headless for automation
python gamry_HiPOZ.py --headless --dates 20250813 20250814 20250815
```

## Command Line Reference

```
python gamry_HiPOZ.py [OPTIONS]

Options:
  --dates DIR [DIR ...]    Specify data directory(ies) to process
  --config FILE            Path to calibration config file (CSV or JSON)
  --gui                    Force GUI mode (even with complete config)
  --headless               Force headless mode (requires complete config)
  --harmonize FILE         Harmonize CSV↔JSON config file and exit

Headless Plotting Options:
  --plot-svsp             Generate S vs P plot
  --plot-bode             Generate Bode plots
  --plot-nyquist          Generate Nyquist plots
  --plot-sigma-conc       Generate conductivity vs concentration plot
  --plot-sigma-temp       Generate conductivity vs temperature plot
  --plot-all              Generate all plots

Examples:
  python gamry_HiPOZ.py
  python gamry_HiPOZ.py --dates 20250815Mahboub2026
  python gamry_HiPOZ.py --headless --dates 20250815
  python gamry_HiPOZ.py --harmonize data/20250815/zAnalysis.csv
```

## Documentation

Comprehensive guides available:

- **[QUICK_START.md](docs/QUICK_START.md)** - Quick reference for common tasks
- **[CALIBRATION_WORKFLOW.md](docs/CALIBRATION_WORKFLOW.md)** - Step-by-step calibration guide
- **[PLOTTING_README.md](docs/PLOTTING_README.md)** - Conductivity plotting guide (σ vs conc, σ vs T, external data)
- **[DIRECTORY_SELECTION_UPDATE.md](docs/DIRECTORY_SELECTION_UPDATE.md)** - New directory selection features
- **[HARMONIZE_COMMAND.md](docs/HARMONIZE_COMMAND.md)** - CSV/JSON synchronization
- **[HEADLESS_MODE.md](docs/HEADLESS_MODE.md)** - Automated batch processing
- **[GUI_CALIBRATION_DISPLAY.md](docs/GUI_CALIBRATION_DISPLAY.md)** - GUI features and tips
- **[FORMAT_HARMONIZATION.md](docs/FORMAT_HARMONIZATION.md)** - Config file format details
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for Claude Code

## Testing

```bash
pip install -e ".[dev]"        # installs pytest

# Speciation and McCleskey model
pytest tests/test_speciation.py -v

# Data, table and plot generation
pytest tests/test_latex_tables.py tests/test_benchtop_data.py \
       tests/test_plot_generation.py

# Individual scripts (not pytest-based)
python tests/test_format_harmonization.py
python mahboub2026/test_mahboub_analysis.py
```

Speciation tests requiring Reaktoro skip cleanly when it is absent, so the suite
runs in a plain pip environment.

**Known pre-existing failures.** `pytest tests/` currently reports 3 failures and
2 collection errors that are unrelated to speciation or the Qt migration; they
also occur on the pre-change code. Additionally,
`tests/test_gui_reorganization.py` **hangs when run headlessly**: its mock
`TimeSeries` lacks an `uncertainties` attribute, so `plot_timeseries` raises and a
*modal* `QMessageBox.critical` (`hipoz_data_selector_gui.py:592`) blocks forever
with no display. Run the other test files, or add the missing attribute to the
mock.

## Troubleshooting

### "cannot parse S or Z" Error
- **Cause:** Standards missing conductivity values
- **Fix:** Add `conductivity_Sm` column values in CSV for standards

### Config File Not Found
- **Cause:** Config in wrong directory
- **Fix:** Config should be in `data/<date>/zAnalysis<date>.csv`

### Plots Not Generating
- **Cause:** Missing TeX installation
- **Fix:** Install LaTeX with required packages (siunitx, upgreek, mhchem)

### Files Showing Wrong P or T
- **Cause:** Metadata in filename or header
- **Fix:** Edit P_MPa and T_K columns in CSV directly

### Want to Skip Bad Measurements
- **Fix:** Mark `exclude` column with `x` in CSV

## Project Structure

```
hipozgenai/
├── gamry_HiPOZ.py              # Main entry point - launches GUI or headless analysis
├── hipoz_data_selector_gui.py # PyQt6 GUI for interactive data curation
├── gamryTools.py              # Core classes (Solution, TimeSeries, circuit fitting)
├── gamryPlots.py              # Publication-quality plotting functions
├── plotting.py                # Low-level publication plot functions
├── study_plots.py             # High-level study plot functions + McCleskey model
├── sigmaElectricMcCleskey2012.py  # McCleskey et al. (2012) conductivity model
├── speciation.py              # WATEQ4F speciation via Reaktoro (optional)
├── validate_speciation.py     # Speciation validation against measurements
├── pyproject.toml             # Dependency declarations (pip install -e .)
├── environment.yml            # Conda environment (incl. conda-only reaktoro)
├── analysis_config.py         # Loads and parses analysis setup files (zAnalysis*.csv/json)
│                              # - Identifies which files are standards vs measurements
│                              # - Stores metadata (P, T, composition, concentrations)
│                              # - Organizes files into calibration groups
│                              # - Used by both GUI and headless mode
├── harmonize_config.py        # CSV↔JSON synchronization tool
├── headless_analysis.py       # Automated batch processing without GUI
├── data/                      # Data directory (user-created)
│   └── <date>/                # One folder per experiment date
│       ├── ConductivityData_Default/  # Gamry instrument output files
│       ├── zAnalysis<date>.csv        # Analysis setup (Excel-friendly)
│       └── zAnalysis<date>.json       # Analysis setup (JSON format)
└── docs/                      # Comprehensive documentation
```

**Key module: `analysis_config.py`**

This module handles the analysis setup file (`zAnalysis*.csv` or `zAnalysis*.json`) which tells HiPOZ:
- Which files are calibration standards (with known conductivity)
- Which files are measurements (unknown conductivity to be determined)
- Metadata for each file: pressure, temperature, composition, concentration
- How to organize files into calibration groups (for bracketed measurements)

The GUI auto-creates this file when you first analyze a dataset. You can then edit it in Excel (CSV) or programmatically (JSON) to add metadata, mark standards, or exclude bad files.

## Contributing

This tool is designed for the planetary science community studying ocean world analogs. Contributions welcome!

## Citation

If you use HiPOZ in your research, please cite:
- The impedance package: https://impedancepy.readthedocs.io

## License

Copyright (c) 2023-24 California Institute of Technology (Caltech). U.S. Government sponsorship acknowledged.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

**The complete GNU General Public License v3.0 text is available in the [LICENSE](LICENSE) file.**

## Contact

**Steven D. Vance**
Jet Propulsion Laboratory, California Institute of Technology
Email: steven.d.vance@jpl.nasa.gov

## Acknowledgments

**Initial Development:** Marshall Styczinski ([@itsmoosh](https://github.com/itsmoosh))

Special thanks to the many students and postdocs of the JPL Habitability and Geophysics Group (HabGeo) for their contributions to testing, validation, and scientific applications.

Developed for high-pressure ocean world analog experiments with support from NASA's Planetary Science Division.
