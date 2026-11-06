# Quick Start: Plotting Cortes & Mahboub Data

## Current Status ✅

You have **two complete plotting workflows**:

### 1. Cortes Gamry-Only (Fast Publication Plots)

```bash
# Generate 5 publication plots with replicate averaging
python3 plot_cortes_publication.py 20250813Cortes 20250814Cortes 20250815Cortes
```

**Outputs**:
- `cortes_plots/single_salts/NaCl_vs_pressure.pdf`
- `cortes_plots/single_salts/NaCl_vs_temperature.pdf`
- `cortes_plots/single_salts/NaCl_vs_concentration.pdf`
- `cortes_plots/mixtures/Mixtures_vs_temperature.pdf`
- `cortes_plots/mixtures/Mixtures_vs_concentration.pdf`

**Features**:
- Averages replicates (69 → 23 data points)
- Calculates SEM error bars
- Combines all concentrations on one plot
- Organized directory structure

### 2. Mahboub Benchtop+Gamry (Full Workflow with McCleskey)

```bash
cd mahboub2026
python3 mahboub2026_plots.py
```

**Outputs**: 10 plots (5 compounds × 2 plot types)
- σ vs concentration with McCleskey model comparison
- σ vs temperature with McCleskey model comparison
- Automatic Gamry high-pressure overlay

## What I Just Added 🆕

### Unified CLI Interface

```bash
# List available studies
python3 plot_study_cli.py --list

# Plot a specific study
python3 plot_study_cli.py mahboub2026
python3 plot_study_cli.py cortes2026

# Filter to specific compounds
python3 plot_study_cli.py mahboub2026 --compounds NaCl MgSO4

# Generate PNG instead of PDF
python3 plot_study_cli.py cortes2026 --format png --dpi 300

# Initialize new study
python3 plot_study_cli.py --init mystudy2026
```

### Batch Processing Script

```bash
# Plot everything
./plot_all.sh

# Plot just Cortes
./plot_all.sh cortes

# Plot just Mahboub
./plot_all.sh mahboub
```

## Architecture Documentation

See `PLOTTING_ARCHITECTURE.md` for:
- Complete data flow diagrams
- Workflow comparisons
- File organization
- GUI integration proposals
- Next steps roadmap

## What Works Right Now

✅ **Cortes Gamry Data**: Ready to plot
✅ **Mahboub Benchtop+Gamry**: Ready to plot
✅ **Replicate Averaging**: Working (`cortes_data_processing.py`)
✅ **McCleskey Model**: Integrated (`study_plots.py`)
✅ **CLI Interface**: Extensible (`plot_study_cli.py`)
✅ **Batch Processing**: Shell script (`plot_all.sh`)

## Next Steps (if you want them)

### Priority 1: Unified CLI
Combine both workflows in `plot_study_cli.py`:
```bash
python3 plot_study_cli.py cortes2026 \
    --workflow gamry-only \
    --average-replicates \
    --combine-concentrations
```

### Priority 2: GUI Integration
Add plotting tab to existing `hipoz_data_selector_gui.py`:
- Study selector
- Compound multi-select
- Workflow options
- Live preview
- Batch export

### Priority 3: Cortes Benchtop Integration
Create full Cortes workflow matching Mahboub pattern:
```bash
python3 cortes2026/cortes2026_plots.py  # Not yet implemented
# Would use: Cortes2026BenchtopData.csv + zAnalysis*.csv
```

## Key Files Reference

```
Plotting Scripts:
├── plot_cortes_publication.py    # Gamry-only (READY)
├── plot_study_cli.py              # Unified CLI (NEW)
└── plot_all.sh                    # Batch wrapper (NEW)

Processing:
├── cortes_data_processing.py     # Replicate averaging
├── gamry_integration.py           # Overlay logic
└── study_plots.py                 # Generalized functions

Study-Specific:
├── mahboub2026/mahboub2026_plots.py     # Full workflow (READY)
└── cortes2026/cortes2026_plots.py       # Stub (needs benchtop data)

Documentation:
├── PLOTTING_ARCHITECTURE.md       # Complete architecture (NEW)
├── CORTES_IMPLEMENTATION_PLAN.md  # Implementation status
└── CORTES_PLOTTING_README.md      # Quick reference
```

## Test It Now

If you have the Python environment ready:

```bash
# Test Cortes plotting (Gamry-only)
python3 plot_cortes_publication.py

# Test batch script
./plot_all.sh cortes

# Test CLI interface
python3 plot_study_cli.py --list
```

## Questions to Guide Next Steps

1. **Do you want the unified CLI to combine both workflows?**
   - Gamry-only (fast, with averaging)
   - Benchtop+Gamry (full, with McCleskey)

2. **GUI priority?**
   - Add to existing DataSelector GUI?
   - Create standalone plotting GUI?
   - CLI is sufficient for now?

3. **Cortes benchtop data integration?**
   - Do you have benchtop data to integrate?
   - Or is Gamry-only sufficient?

4. **Plot customization needs?**
   - Current styling OK?
   - Want interactive customization?
   - Preset templates needed?

Let me know what direction you'd like to take!
