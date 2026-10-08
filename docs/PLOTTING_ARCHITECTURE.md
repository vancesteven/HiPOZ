# HiPOZ Plotting Architecture

## Overview

The HiPOZ project has evolved two complementary plotting approaches:

1. **Gamry-only workflow**: Fast publication plots from impedance data
2. **Benchtop+Gamry workflow**: Full integration with McCleskey model comparison

## Current Architecture

```
Data Sources
│
├── Gamry Impedance Data
│   └── data/*/zAnalysis*.csv
│       - High-pressure measurements
│       - Circuit fitting results
│       - Conductivity calculated
│
└── Benchtop Probe Data
    └── */Benchtop Data.csv
        - Low-pressure calibration
        - Temperature series
        - Concentration series

Processing Modules
│
├── cortes_data_processing.py
│   - load_cortes_data()
│   - average_replicates()
│   - separate_by_composition()
│
├── gamry_integration.py
│   - load_gamry_results()
│   - extract_compound_overlay()
│
└── study_plots.py
    - plot_study_concentration()
    - plot_study_temperature()
    - compute_mccleskey_model()

Plotting Scripts
│
├── plot_cortes_publication.py
│   Purpose: Gamry-only publication plots
│   Input: zAnalysis*.csv
│   Output: 5 plots (NaCl P/T/conc + mixtures T/conc)
│   Pattern: Replicate averaging → combined plots
│
├── cortes2026_plots.py
│   Purpose: Benchtop + Gamry overlay
│   Input: Cortes2026BenchtopData.csv + zAnalysis*.csv
│   Output: Compound-specific plots with McCleskey
│   Pattern: Mahboub2026 style
│
├── plot_study_cli.py
│   Purpose: Unified CLI for any study
│   Features: Study registry, auto-detection, formats
│   Pattern: Extensible configuration
│
└── plot_all.sh
    Purpose: Batch plotting wrapper
    Usage: ./plot_all.sh [study] [format]
```

## Workflow Comparison

### Mahboub Pattern (Benchtop + Gamry)

```python
# Load benchtop data from CSV
benchtop_data = load_study_data('Mahboub2026BenchtopData.csv')

# Load Gamry overlay from impedance analysis
gamry_df = load_gamry_results('data/20250815Mahboub2026')
gamry_overlay = extract_compound_overlay(gamry_df, 'NaCl')

# Generate compound-specific plot with McCleskey
plot_study_concentration(
    data=benchtop_data,
    compound='NaCl',
    gamry_data=gamry_overlay,     # Overlay high-P data
    show_delta=True,               # Show McCleskey comparison
    output_file='nacl_vs_conc.pdf'
)
```

**Output**: Separate plots per compound (NaCl, MgSO4, NH4Cl, etc.)
**Features**: McCleskey model, Δ% subplot, publication formatting

### Cortes Pattern (Gamry-only)

```python
# Load Gamry data and average replicates
data = load_cortes_data(['20250813Cortes', '20250814Cortes'])
averaged = average_replicates(data)  # 69 → 23 points

# Separate by composition
separated = separate_by_composition(averaged)
# → single_salts: NaCl at 0.5M, 0.75M, 2M
# → mixtures: NaCl+MgSO4 at various ratios

# Generate combined plots
plot_sigma_vs_pressure_combined(
    data_list=separated['single_salts'],
    compound_type='NaCl',
    output_file='NaCl_vs_pressure.pdf'
)
```

**Output**: Combined plots (all concentrations on one figure)
**Features**: Replicate averaging (mean ± SEM), organized directories

## Data Flow Diagrams

### Gamry-Only (Current Cortes)

```
Raw Gamry Files (*.txt)
    ↓
gamry_HiPOZ.py (circuit fitting)
    ↓
zAnalysis*.csv (conductivity + uncertainty)
    ↓
cortes_data_processing.py (averaging)
    ↓
plot_cortes_publication.py
    ↓
5 Publication PDFs
```

### Benchtop+Gamry (Mahboub Pattern)

```
Benchtop CSV              Gamry zAnalysis CSV
    ↓                           ↓
load_study_data()      load_gamry_results()
    ↓                           ↓
benchtop_data          extract_compound_overlay()
    └───────────┬───────────────┘
                ↓
    plot_study_concentration()
    plot_study_temperature()
                ↓
        10 Publication PDFs
    (one per compound × 2 plot types)
```

## File Locations

```
hipozgenai/
├── Plot Generation
│   ├── plot_cortes_publication.py       # Gamry-only (5 plots)
│   ├── plot_gamry_cortes.py             # Quick Gamry visualization
│   ├── plot_study_cli.py                # Unified CLI interface
│   └── plot_all.sh                      # Batch wrapper
│
├── Data Processing
│   ├── cortes_data_processing.py        # Replicate averaging
│   ├── gamry_integration.py             # Gamry overlay logic
│   └── study_plots.py                   # Generalized plotting
│
├── Configuration
│   └── config_plots.py                  # Universal styling
│
├── Study-Specific
│   ├── mahboub2026/
│   │   ├── mahboub2026_plots.py         # Full benchtop+Gamry
│   │   └── Mahboub2026BenchtopData.csv
│   │
│   └── cortes2026/
│       ├── cortes2026_plots.py          # Full benchtop+Gamry
│       ├── Cortes2026BenchtopData.csv
│       └── parse_benchtop_data.py
│
└── Data
    ├── data/20250813Cortes/
    │   └── zAnalysis20250813.csv
    ├── data/20250814Cortes/
    │   └── zAnalysis20250814.csv
    └── data/20250815Cortes/
        └── zAnalysis20250815.csv
```

## Usage Examples

### Quick Gamry Visualization

```bash
# Simple 3-plot overview
python3 plot_gamry_cortes.py 20250813Cortes 20250814Cortes
# Output: cortes_plots/sigma_vs_{pressure,temperature,concentration}.pdf
```

### Publication Plots (Gamry-only)

```bash
# 5 publication-quality plots with replicate averaging
python3 plot_cortes_publication.py 20250813Cortes 20250814Cortes 20250815Cortes
# Output: cortes_plots/single_salts/ + cortes_plots/mixtures/
```

### Full Workflow (Benchtop + Gamry)

```bash
# Compound-specific plots with McCleskey overlay
cd mahboub2026
python3 mahboub2026_plots.py
# Output: mahboub_plots/*.pdf (10 plots)
```

### Unified CLI

```bash
# Use study registry
python3 plot_study_cli.py mahboub2026
python3 plot_study_cli.py cortes2026

# Custom configuration
python3 plot_study_cli.py --init newstudy2026
python3 plot_study_cli.py newstudy2026 --config newstudy2026_config.json
```

### Batch Processing

```bash
# Plot everything
./plot_all.sh

# Study-specific
./plot_all.sh cortes
./plot_all.sh mahboub
```

## Next Steps for Generalized CLI/GUI

### Phase 1: CLI Enhancement

**Goal**: Unify Gamry-only and benchtop+Gamry workflows

**Implementation**:
```python
# plot_study_cli.py enhancements
parser.add_argument('--workflow', choices=['gamry-only', 'benchtop+gamry', 'auto'])
parser.add_argument('--average-replicates', action='store_true')
parser.add_argument('--combine-concentrations', action='store_true')
parser.add_argument('--show-model', action='store_true')
```

**Usage**:
```bash
# Gamry-only with averaging (Cortes pattern)
python3 plot_study_cli.py cortes2026 \
    --workflow gamry-only \
    --average-replicates \
    --combine-concentrations

# Full workflow with McCleskey (Mahboub pattern)
python3 plot_study_cli.py cortes2026 \
    --workflow benchtop+gamry \
    --show-model
```

### Phase 2: GUI Integration Options

#### Option A: Extend Existing DataSelector GUI

```python
# Add to hipoz_data_selector_gui.py
class PlottingTab(QWidget):
    """New tab for plot generation."""

    def __init__(self, parent):
        # Study selection
        self.study_combo = QComboBox(['Mahboub2026', 'Cortes2026'])

        # Workflow selection
        self.workflow_combo = QComboBox(['Gamry-only', 'Benchtop+Gamry'])

        # Compound selection (multi-select)
        self.compound_list = QListWidget()
        self.compound_list.setSelectionMode(QAbstractItemView.MultiSelection)

        # Options
        self.average_replicates_check = QCheckBox('Average replicates')
        self.show_model_check = QCheckBox('Show McCleskey model')
        self.combine_check = QCheckBox('Combine concentrations')

        # Generate button
        self.generate_button = QPushButton('Generate Plots')
        self.generate_button.clicked.connect(self.generate_plots)

    def generate_plots(self):
        """Generate plots with selected options."""
        study = self.study_combo.currentText()
        workflow = self.workflow_combo.currentText()
        compounds = [item.text() for item in self.compound_list.selectedItems()]

        # Call appropriate plotting function
        if workflow == 'Gamry-only':
            plot_gamry_only(study, compounds, 
                          average=self.average_replicates_check.isChecked())
        else:
            plot_benchtop_gamry(study, compounds,
                              show_model=self.show_model_check.isChecked())
```

#### Option B: Standalone Plotting GUI

```python
# new file: plot_gui.py
class PlotGeneratorWindow(QMainWindow):
    """Standalone plotting interface."""

    def __init__(self):
        # Left panel: Study configuration
        # - Study selector
        # - Data directory browser
        # - Compound detection
        # - Workflow selector

        # Center panel: Plot preview
        # - Matplotlib canvas
        # - Zoom/pan controls
        # - Plot type selector

        # Right panel: Export options
        # - Format (PDF/PNG/both)
        # - DPI slider
        # - Batch export
        # - Progress bar

        # Bottom: Log output
```

**Launch**:
```bash
python3 plot_gui.py
```

### Phase 3: Interactive Features

#### Real-time Preview
- Live plot updates as options change
- McCleskey model toggle
- Colormap selection
- Font size adjustment

#### Batch Processing
- Select multiple compounds → generate all plots
- Progress bar and ETA
- Error recovery

#### Export Templates
- Save plot configurations as templates
- "Mahboub style", "Cortes style" presets
- Custom templates

#### Data Curation Integration
- Direct link from DataSelector to plotting
- "Plot selected" button
- Sync compound selection

## Configuration Files

### Study Configuration (JSON)

```json
{
  "name": "Cortes et al. (2026)",
  "benchtop_csv": "cortes2026/Cortes2026BenchtopData.csv",
  "gamry_dirs": [
    "data/20250813Cortes",
    "data/20250814Cortes",
    "data/20250815Cortes"
  ],
  "output_dir": "cortes2026/cortes_plots",
  "compounds": "auto",
  "workflows": {
    "gamry_only": {
      "average_replicates": true,
      "combine_concentrations": true,
      "p_tolerance": 2.0,
      "t_tolerance": 0.5
    },
    "benchtop_gamry": {
      "show_delta": "auto",
      "mccleskey_compounds": ["NaCl", "KCl", "MgSO4"]
    }
  },
  "latex_names": {
    "MgSO4": "MgSO$_4$",
    "NaCl:MgSO4_1:1": "NaCl:MgSO$_4$ (1:1)"
  }
}
```

### Plot Style Configuration

```python
# config_plots.py enhancements
PLOT_STYLES = {
    'publication': {
        'figsize': (10, 7),
        'dpi': 300,
        'fontsize_label': 14,
        'fontsize_title': 16,
        'fontsize_legend': 10,
        'linewidth': 2,
        'markersize': 8
    },
    'presentation': {
        'figsize': (12, 8),
        'dpi': 150,
        'fontsize_label': 18,
        'fontsize_title': 22,
        'fontsize_legend': 14,
        'linewidth': 3,
        'markersize': 10
    },
    'quick': {
        'figsize': (8, 6),
        'dpi': 100,
        'fontsize_label': 12,
        'fontsize_title': 14,
        'fontsize_legend': 9,
        'linewidth': 1.5,
        'markersize': 6
    }
}
```

## Design Principles

1. **Separation of Concerns**
   - Data loading separate from processing
   - Processing separate from plotting
   - Configuration separate from implementation

2. **Workflow Flexibility**
   - Support both Gamry-only and benchtop+Gamry
   - Allow replicate averaging or individual points
   - Enable combined or separate plots per compound

3. **Extensibility**
   - Easy to add new studies
   - Pluggable data loaders
   - Customizable plot styles

4. **User Experience**
   - Simple defaults that work
   - Progressive disclosure of options
   - Clear error messages
   - Informative progress output

## Summary

The current architecture provides:
- ✅ Gamry-only publication plots (Cortes)
- ✅ Benchtop+Gamry overlay plots (Mahboub)
- ✅ Generalized plotting functions
- ✅ CLI interface foundation
- ✅ Batch processing scripts

Still needed:
- [ ] Unified CLI combining both workflows
- [ ] GUI plotting interface
- [ ] Interactive plot preview
- [ ] Template/preset system
- [ ] Integration with DataSelector GUI

All the building blocks exist - the next step is integration!
