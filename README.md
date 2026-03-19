# HiPOZ

**H**igh-**P**ressure **O**cean world analog impedance (**Z**) analyzer

Software for measuring and analyzing electrical impedance spectroscopy data from high-pressure fluid experiments.

## Quick Start

### For Students (No Python Editing Required!)

```bash
# Run with GUI - will prompt for directory selection:
python gamry_HiP.py

# Or specify directory from command line:
python gamry_HiP.py --dates 20250815Mahboub2026
```

That's it! Select your data directory and the GUI will guide you through calibration.

### For Automated Analysis

```bash
# Headless mode (requires config file with standards):
python gamry_HiP.py --headless --dates 20250815Mahboub2026

# Harmonize CSV and JSON after editing:
python gamry_HiP.py --harmonize data/20250815/zAnalysis20250815.csv
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

### Required Dependencies

Install with conda/mamba:
```bash
mamba install numpy matplotlib
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
python gamry_HiP.py
```
- Dialog opens to select data directory
- GUI loads all measurements
- Follow on-screen instructions to mark standards and associate measurements

**Option B: Command Line**
```bash
python gamry_HiP.py --dates 20250815Mahboub2026
```

**Option C: Headless Mode (Automated)**
```bash
# First run: Create config with GUI
python gamry_HiP.py --dates 20250815

# Edit CSV in Excel to mark standards and add conductivity values

# Subsequent runs: Automated analysis
python gamry_HiP.py --headless --dates 20250815
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
python gamry_HiP.py --harmonize data/20250815/zAnalysis20250815.csv
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

## Common Workflows

### Workflow 1: First Time Analysis
```bash
# 1. Run GUI
python gamry_HiP.py --dates 20250815

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
python gamry_HiP.py --harmonize data/20250815/zAnalysis20250815.csv

# 5. Re-run analysis with updated metadata
python gamry_HiP.py --dates 20250815
```

### Workflow 3: Batch Processing
```bash
# Process multiple days at once
python gamry_HiP.py --dates 20250813 20250814 20250815

# Or headless for automation
python gamry_HiP.py --headless --dates 20250813 20250814 20250815
```

## Command Line Reference

```
python gamry_HiP.py [OPTIONS]

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
  --plot-all              Generate all plots

Examples:
  python gamry_HiP.py
  python gamry_HiP.py --dates 20250815Mahboub2026
  python gamry_HiP.py --headless --dates 20250815
  python gamry_HiP.py --harmonize data/20250815/zAnalysis.csv
```

## Documentation

Comprehensive guides available:

- **[QUICK_START.md](QUICK_START.md)** - Quick reference for common tasks
- **[CALIBRATION_WORKFLOW.md](CALIBRATION_WORKFLOW.md)** - Step-by-step calibration guide
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
├── gamry_HiP.py              # Main entry point
├── DataSelector.py            # PyQt5 GUI for data curation
├── gamryTools.py              # Core classes (Solution, TimeSeries)
├── gamryPlots.py              # Plotting functions
├── calibration_config.py      # Config file handling
├── harmonize_config.py        # CSV↔JSON synchronization
├── headless_analysis.py       # Automated batch processing
├── data/                      # Data directory
│   └── <date>/                # One folder per experiment date
│       ├── ConductivityData_Default/  # Gamry output files
│       ├── zAnalysis<date>.csv        # Config file (Excel-friendly)
│       └── zAnalysis<date>.json       # Config file (JSON)
└── docs/                      # Comprehensive documentation
```

## Contributing

This tool is designed for the planetary science community studying ocean world analogs. Contributions welcome!

## Citation

If you use HiPOZ in your research, please cite:
- The impedance package: https://impedancepy.readthedocs.io
- Pan et al. (2021) cell model: https://doi.org/10.1029/2021GL094020

## License

[Add your license here]

## Contact

[Add contact information]

## Acknowledgments

Developed for high-pressure ocean world analog experiments. Special thanks to the planetary science community for feedback and testing.
