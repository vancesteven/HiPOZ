# HiPOZ Documentation

Complete documentation for the HiPOZ impedance analysis system.

## Quick Links

### User Guides
- **[Main README](../README.md)** - Project overview and getting started
- **[Quick Start Guide](QUICK_START.md)** - Daily analysis workflow
- **[GUI Overview](GUI_OVERVIEW.md)** - Data selector interface and features
- **[GUI Architecture](GUI_ARCHITECTURE.md)** - Complete GUI implementation details
- **[Calibration Guide](CALIBRATION.md)** - How to calibrate and analyze impedance data
- **[Plotting Guide](PLOTTING.md)** - How to create publication plots
- **[Plotting Features](PLOTTING_FEATURES.md)** - Detailed plotting function reference (NEW)

### Developer Guides
- **[Testing Guide](TESTING.md)** - How to run and write tests
- **[Test Suite Reference](../tests/README.md)** - Complete test documentation
- **[Mahboub 2026 Study](../mahboub2026/README.md)** - Example study documentation
- **[Cortes 2026 Study](../cortes2026/README.md)** - Another example study

## Documentation Files

### QUICK_START.md
Daily analysis workflow guide:
- Auto-creating config files with GUI
- Editing zAnalysis CSV files in Excel
- Command reference for common operations
- Configuration file formats (CSV/JSON)
- Troubleshooting config file issues

**When to use:** Starting daily data analysis or setting up new experiments.

### GUI_OVERVIEW.md
Complete Data Selector GUI reference:
- Tab structure and navigation
- Data table columns and editing
- Workflow operations (Mark Standard, Associate Measurements)
- Configuration file management
- Keyboard shortcuts and advanced features

**When to use:** Learning GUI features or customizing workflow.

### CALIBRATION.md
Complete guide to the calibration workflow:
- Creating calibration config files (CSV/JSON)
- Defining standards with known conductivity
- Associating measurements with calibration groups
- Running headless analysis
- Troubleshooting common issues

**When to use:** Setting up a new experiment or analyzing impedance data.

### PLOTTING.md
Guide to the publication plotting system:
- Loading benchtop data from CSV
- Integrating Gamry impedance overlays
- Customizing plot appearance (fonts, colors, styles)
- Creating study-specific plotting scripts
- Reusing functions for multiple studies

**When to use:** Creating publication-quality conductivity plots.

### GUI_ARCHITECTURE.md
Complete GUI implementation documentation:
- Current tab structure and layout system
- Proposed tab-based redesign for future enhancement
- Data table columns and auto-conversion logic
- Multi-component solution handling
- Calibration workflow and state management
- Extension points for adding features

**When to use:** Understanding GUI internals or planning modifications.

### PLOTTING_FEATURES.md
Detailed plotting function reference (NEW):
- `plot_conductivity_vs_temperature()` - σ vs T with Arrhenius analysis
- `plot_conductivity_vs_molality()` - σ vs m concentration dependence
- Timeseries, Bode, Nyquist, and S vs P plots
- LaTeX table generation for publications
- Export options (PDF, PNG, SVG)
- Customization and styling examples

**When to use:** Creating custom plots or understanding plotting functions.

### TESTING.md
Guide to testing HiPOZ code:
- Running the test suite
- Writing new tests
- Test coverage and validation
- Pre-commit testing workflow
- Continuous integration

**When to use:** Modifying core code or adding new features.

## Study-Specific Documentation

### Mahboub et al. (2026)
Location: `../mahboub2026/README.md`

Complete documentation for the Mahboub conductivity study:
- 10 publication plots (5 compounds × 2 plot types)
- Benchtop + Gamry impedance integration
- Frozen sample handling
- Error handling and troubleshooting

**Example of:** How to document a complete study.

## Contributing Documentation

When adding new features:

1. **User-facing changes:** Update relevant guide in `docs/`
2. **New features:** Add section to appropriate guide
3. **Breaking changes:** Update main README.md
4. **Study-specific:** Create study folder with README.md

Keep documentation:
- ✅ Clear and concise
- ✅ Example-driven
- ✅ Up-to-date with code
- ✅ Well-organized

## Documentation Standards

### File Naming
- Use `.md` extension
- Use UPPERCASE for general docs (CALIBRATION.md)
- Use lowercase for specific docs (mahboub2026/README.md)

### Structure
Each guide should have:
1. **Overview** - What this document covers
2. **Quick Start** - Minimal example
3. **Detailed Guide** - Step-by-step instructions
4. **Troubleshooting** - Common issues
5. **Examples** - Real-world usage

### Cross-References
- Use relative links: `[Main README](../README.md)`
- Link to related docs at the end
- Keep links up-to-date when moving files

## Getting Help

- **Questions about usage:** Check relevant guide above
- **Bug reports:** See main README for issue tracker
- **Feature requests:** Discuss before implementing
- **Code questions:** See inline comments and docstrings

## See Also

- [Main README](../README.md) - Project overview
- [Tests Directory](../tests/) - Test suite
- [Mahboub Study](../mahboub2026/) - Example analysis
