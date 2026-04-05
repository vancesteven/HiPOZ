# HiPOZ Documentation Index

**Last Updated**: April 4, 2026

Complete guide to all documentation in this repository.

## Quick Start

**New Users**: Start here
- **README.md** - Project overview and installation
- **CLAUDE.md** - Claude Code instructions and environment setup
- **QUICK_START_PLOTTING.md** - How to generate plots quickly

## Plotting Documentation

### Main Guides

1. **MCCLESKEY_CORTES_GUIDE.md** (400 lines) ⭐
   - Complete usage guide for Cortes plotting
   - Command-line options and examples
   - McCleskey model comparison
   - Troubleshooting and FAQ
   - **Status**: ✅ Current (v2.0)

2. **PLOTTING_ENHANCEMENTS_SUMMARY.md** (600 lines)
   - Pressure colormap implementation
   - LaTeX table generation
   - McCleskey comparison verification
   - Usage examples for publications
   - **Status**: ✅ Current

3. **MCCLESKEY_IMPLEMENTATION_SUMMARY.md** (378 lines)
   - Technical implementation details
   - Architecture and data flow
   - Code statistics and file structure
   - Verification checklist
   - **Status**: ✅ Current (v2.0)

4. **PLOT_FILTERING_UPDATE.md**
   - NaCl 0.5M filtering rationale
   - Mixture concentration plot addition
   - Plot inventory updates
   - **Status**: ✅ Current

5. **PLOTTING_ARCHITECTURE.md**
   - System architecture overview
   - Module relationships
   - Design patterns
   - **Status**: ✅ Current

6. **PLOTTING_FORMATTING_UPDATES.md**
   - Formatting standards
   - Marker-only style
   - LaTeX configuration
   - Colormap choices
   - **Status**: ✅ Current

7. **docs/GUI_CHANGES.md** ⭐ (NEW)
   - Performance improvements
   - McCleskey model integration
   - User experience enhancements
   - Button color indicators
   - **Status**: ✅ Current (April 2026)

### Specific Features

8. **PLOT_GENERATION_20250815.md**
   - Integration of 20250815 dataset
   - Data quality notes
   - Technical issues resolved
   - **Status**: ✅ Current

9. **PLOT_FORMATTING_COMPLETE.md**
   - Completion summary for formatting updates
   - Test results (25/25 passing)
   - **Status**: ✅ Archive (historical)

## Development Documentation

### Planning and Tasks

10. **TODO_LIST.md** ⭐
    - All pending development tasks
    - Prioritized by urgency
    - Detailed implementation plans
    - GUI enhancement roadmap
    - **Status**: ✅ Current (active development)

11. **CORTES_IMPLEMENTATION_PLAN.md**
    - Original Cortes plotting plan
    - Requirements and specifications
    - **Status**: ✅ Archive (completed)

12. **CORTES_PLOTTING_README.md**
    - Early Cortes plotting documentation
    - **Status**: ⚠️ Superseded by MCCLESKEY_CORTES_GUIDE.md

### Cleanup and Organization

13. **CLEANUP_PLAN.md**
    - File cleanup analysis
    - Recommendations for removal
    - **Status**: ✅ Archive (cleanup completed)

14. **CLEANUP_EXECUTED.md**
    - Cleanup execution summary
    - Files moved to archive/
    - **Status**: ✅ Archive

15. **READY_FOR_COMMIT.md**
    - Pre-commit checklist
    - Files ready for git commit
    - Code quality verification
    - **Status**: ✅ Current

16. **COMMIT_MESSAGE.md**
    - Prepared git commit messages
    - Detailed change descriptions
    - **Status**: ✅ Current

## Technical Documentation

### Data Processing

17. **CALIBRATION.md**
    - Calibration workflow documentation
    - Cell constant determination
    - Configuration file format
    - **Status**: ✅ Current

18. **docs/** directory
    - Additional technical documentation
    - API references (if present)

### Configuration

19. **analysis_config_example.json**
    - Example configuration file
    - JSON format specification

20. **analysis_config_example.csv**
    - Example CSV configuration
    - CSV format specification

## Reference Materials

### External Documents

21. **mahboub2026/supplement.tex**
    - Mahboub LaTeX supplement template
    - Reference for table formatting
    - **Status**: ✅ Reference

22. **mahboub2026/Mahboub2026rev2Supplement.pdf**
    - Published supplement PDF
    - Data exclusion rationale

23. **precision_formatting_report.md**
    - Numerical precision documentation
    - Formatting standards

## Documentation by Topic

### 📊 Plotting

**Getting Started**:
1. QUICK_START_PLOTTING.md
2. MCCLESKEY_CORTES_GUIDE.md
3. docs/GUI_CHANGES.md

**Advanced**:
4. PLOTTING_ENHANCEMENTS_SUMMARY.md
5. PLOTTING_ARCHITECTURE.md

**Reference**:
6. MCCLESKEY_IMPLEMENTATION_SUMMARY.md
7. PLOT_FILTERING_UPDATE.md

### 🔧 Development

**Active Work**:
1. TODO_LIST.md (all pending tasks)
2. READY_FOR_COMMIT.md (current status)

**Planning**:
3. CORTES_IMPLEMENTATION_PLAN.md (historical)

### 📝 Configuration

1. CALIBRATION.md (workflow)
2. analysis_config_example.json (template)
3. CLAUDE.md (environment)

### 📚 Reference

1. mahboub2026/supplement.tex (LaTeX template)
2. PLOTTING_FORMATTING_UPDATES.md (standards)

## Status Legend

- ✅ **Current** - Up-to-date and actively maintained
- ⚠️ **Superseded** - Replaced by newer documentation
- 🗄️ **Archive** - Historical, completed, or deprecated
- ⭐ **Essential** - Start here for this topic

## Documentation Hierarchy

```
HiPOZ Documentation
│
├── Getting Started
│   ├── README.md
│   ├── CLAUDE.md
│   └── QUICK_START_PLOTTING.md
│
├── Plotting (Main Topic)
│   ├── User Guides
│   │   ├── MCCLESKEY_CORTES_GUIDE.md ⭐
│   │   ├── PLOTTING_ENHANCEMENTS_SUMMARY.md
│   │   └── PLOT_FILTERING_UPDATE.md
│   │
│   ├── Technical
│   │   ├── MCCLESKEY_IMPLEMENTATION_SUMMARY.md
│   │   ├── PLOTTING_ARCHITECTURE.md
│   │   └── PLOTTING_FORMATTING_UPDATES.md
│   │
│   └── Specific Features
│       ├── PLOT_GENERATION_20250815.md
│       └── PLOT_FORMATTING_COMPLETE.md
│
├── Development
│   ├── TODO_LIST.md ⭐
│   ├── READY_FOR_COMMIT.md
│   ├── COMMIT_MESSAGE.md
│   └── CORTES_IMPLEMENTATION_PLAN.md
│
├── Configuration
│   ├── CALIBRATION.md
│   ├── analysis_config_example.json
│   └── CLAUDE.md
│
└── Reference
    ├── mahboub2026/supplement.tex
    └── PLOTTING_FORMATTING_UPDATES.md
```

## Finding Documentation

### By Task

**"I want to generate plots"**
→ MCCLESKEY_CORTES_GUIDE.md

**"I want to add McCleskey comparisons"**
→ MCCLESKEY_CORTES_GUIDE.md, Section "With McCleskey Comparison"

**"I want to create LaTeX tables"**
→ PLOTTING_ENHANCEMENTS_SUMMARY.md, Section "LaTeX Data Tables"

**"I want to customize plot appearance"**
→ PLOTTING_FORMATTING_UPDATES.md

**"I want to understand GUI changes"**
→ docs/GUI_CHANGES.md

**"I want to understand the architecture"**
→ PLOTTING_ARCHITECTURE.md

**"I want to know what's planned next"**
→ TODO_LIST.md

**"I want to calibrate impedance data"**
→ CALIBRATION.md

### By File Type

**Markdown (.md)** - Human-readable documentation
**LaTeX (.tex)** - Publication templates
**JSON** - Configuration file templates
**CSV** - Data file templates

## Maintenance

### Updating Documentation

When making changes:
1. Update relevant guide (MCCLESKEY_CORTES_GUIDE.md, etc.)
2. Update MCCLESKEY_IMPLEMENTATION_SUMMARY.md if architecture changes
3. Update TODO_LIST.md if adding/completing tasks
4. Update READY_FOR_COMMIT.md before commits
5. Update this index if adding new documents

### Documentation Standards

- **Headings**: Use descriptive, hierarchical headings
- **Code blocks**: Always specify language (```python, ```bash)
- **Status**: Include date and status at top of each document
- **Examples**: Include working examples for all features
- **Cross-references**: Link to related documents

## Contributing

When adding new documentation:
1. Follow existing formatting standards
2. Add entry to this index
3. Update related documents with cross-references
4. Include status and date
5. Add to appropriate hierarchy section

---

**Index Version**: 1.1
**Last Reviewed**: April 4, 2026
**Next Review**: April 11, 2026
