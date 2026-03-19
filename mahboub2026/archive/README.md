# Mahboub 2026 Archive

This directory contains historical versions of Mahboub analysis scripts before generalization and refactoring.

## Files

### `MahboubEtAl2026.py` (40 KB, 2026-02-13)
**Original plotting script** before generalization.

- Hardcoded data arrays instead of CSV files
- Custom plotting functions (not reusable)
- All code in single file (~1200 lines)
- No error handling
- No testing

**Replaced by:** Modular system with `mahboub2026_plots.py` + `gamry_integration.py` + `study_plots.py`

### `mahboub2026_plots.py` (12 KB, 2026-03-19 11:31)
**Intermediate version** from root directory before final refactoring.

- Used CSV data files ✓
- Inline Gamry loading code (~65 lines)
- Inline error handling (~130 lines)
- Inline overlay extraction (~40 lines per compound)
- Total: ~420 lines

**Replaced by:** Refactored version using `gamry_integration.py` functions (~340 lines)

### `mahboub2026_data.py` (11 KB, 2026-03-19)
**Data processing script** from before CSV workflow.

- Converted hardcoded arrays to CSV format
- One-time use for data migration
- No longer needed (CSV files exist)

## Current Version

The current analysis system is in the parent directory (`mahboub2026/`):

```
mahboub2026/
├── mahboub2026_plots.py          # Current plotting script (refactored)
├── Mahboub2026BenchtopData.csv   # Benchtop data
├── test_error_handling.py        # Comprehensive test suite
├── README.md                     # Current documentation
└── archive/                      # This directory (historical files)
```

Parent directory (`hipozgenai/`) contains generalized modules:
- `gamry_integration.py` - Reusable Gamry functions
- `study_plots.py` - Reusable plotting functions
- `config_plots.py` - Universal configuration

## Evolution Timeline

1. **2026-02-13:** `MahboubEtAl2026.py` created
   - Original monolithic script
   - Hardcoded data
   - Mahboub-specific everything

2. **2026-03-19:** CSV migration (`mahboub2026_data.py`)
   - Extracted data to `Mahboub2026BenchtopData.csv`
   - Created initial `mahboub2026_plots.py`

3. **2026-03-19:** Generalization
   - Extracted reusable functions to `plotting.py`, `study_plots.py`
   - Created `config_plots.py` for universal settings
   - Moved to `mahboub2026/` subdirectory

4. **2026-03-19:** Gamry integration refactoring
   - Created `gamry_integration.py` module
   - Refactored `mahboub2026_plots.py` to use generalized functions
   - Added comprehensive test suite
   - **Current version: 340 lines vs original 1200+ lines**

## Why Keep Archive?

- Historical reference
- Shows evolution of analysis workflow
- Can revert if needed (though current version is superior)
- Documents lessons learned

## Do Not Use These Files

These are **archived for reference only**. They are:
- ❌ Not maintained
- ❌ No error handling
- ❌ Not tested
- ❌ Not compatible with current workflow

**Use the current version instead:** `../mahboub2026_plots.py`

## See Also

- `../README.md` - Current Mahboub analysis documentation
- `../REFACTORING_SUMMARY.md` - Details of generalization process
- `../ERROR_HANDLING.md` - Error handling documentation
- `../../gamry_integration.py` - Current Gamry integration module
