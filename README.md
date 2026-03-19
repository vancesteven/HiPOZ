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
- Interactive data curation with PyQt5 GUI
- Calibration with KCl standards
- Cell constant calculation and conductivity determination
- Export to CSV with uncertainty quantification
- Generate publication-quality plots (Nyquist, Bode, S vs P)

## Installation

### Version Requirements

**Python Version:**
- **Required: Python 3.11** (current production version)
- Python 3.12+ not yet tested with PlanetProfile and SeaFreeze dependencies
- Python 3.10 may work but not officially supported

**Critical Package Versions:**
- **numpy 1.26.4** (required - compatibility with impedance package)
- Later numpy versions (2.x) may cause issues with impedance fitting

**Why Python 3.11?**

We currently use Python 3.11 for maximum compatibility with PlanetProfile and SeaFreeze. While newer Python versions (3.12, 3.13) offer performance improvements and better error messages, upgrading would require:
- Testing PlanetProfile compatibility
- Testing SeaFreeze compatibility
- Verifying impedance package stability
- Updating CI/CD workflows

**Python 3.12+ advantages** (for future consideration):
- ✨ 5-10% performance improvements
- ✨ Better error messages with syntax highlighting
- ✨ Improved type hinting features
- ✨ PEP 701: Better f-string syntax
- ✨ Lower memory overhead

We plan to evaluate Python 3.12+ once dependencies are tested and stable.

### Required Dependencies

Install with conda/mamba:
```bash
mamba install "numpy==1.26.4" matplotlib
```

Install with conda-forge:
```bash
mamba install -c conda-forge schemdraw
```

Install with pip:
```bash
pip install impedance SeaFreeze PlanetProfile bidict PyQt5 tqdm
```

**Important:** A working TeX installation is required for Matplotlib rendering. The plotting system uses LaTeX for figure labels and requires STIX fonts, siunitx, upgreek, and mhchem packages.

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

- **[QUICK_START.md](QUICK_START.md)** - Quick reference for common tasks
- **[CALIBRATION_WORKFLOW.md](CALIBRATION_WORKFLOW.md)** - Step-by-step calibration guide
- **[PLOTTING_README.md](PLOTTING_README.md)** - Conductivity plotting guide (σ vs conc, σ vs T, external data)
- **[DIRECTORY_SELECTION_UPDATE.md](DIRECTORY_SELECTION_UPDATE.md)** - New directory selection features
- **[HARMONIZE_COMMAND.md](HARMONIZE_COMMAND.md)** - CSV/JSON synchronization
- **[HEADLESS_MODE.md](HEADLESS_MODE.md)** - Automated batch processing
- **[GUI_CALIBRATION_DISPLAY.md](GUI_CALIBRATION_DISPLAY.md)** - GUI features and tips
- **[FORMAT_HARMONIZATION.md](FORMAT_HARMONIZATION.md)** - Config file format details
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for Claude Code

## Testing

Run test suites:
```bash
# Test format harmonization
python test_format_harmonization.py

# Test analysis workflow
python test_mahboub_analysis.py
```

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
├── DataSelector.py            # PyQt5 GUI for interactive data curation
├── gamryTools.py              # Core classes (Solution, TimeSeries, circuit fitting)
├── gamryPlots.py              # Publication-quality plotting functions
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
