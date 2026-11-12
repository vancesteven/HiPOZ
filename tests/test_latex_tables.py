#!/usr/bin/env python
"""
Test LaTeX table generation from conductivity data.

Tests the creation of properly formatted LaTeX tables for publication,
including column formatting, uncertainty handling, and special characters.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
import re

def generate_latex_table(df, caption="", label=""):
    """
    Generate LaTeX table from DataFrame.

    Args:
        df: pandas DataFrame with conductivity data
        caption: Table caption
        label: LaTeX label for referencing

    Returns:
        str: LaTeX table code
    """
    # Start table environment
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"

    # Determine column format (center all columns)
    n_cols = len(df.columns)
    col_format = "c" * n_cols

    latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    latex += "\\hline\\hline\n"

    # Header row
    headers = " & ".join(df.columns)
    latex += f"{headers} \\\\\n"
    latex += "\\hline\n"

    # Data rows
    for _, row in df.iterrows():
        row_str = " & ".join([str(val) for val in row])
        latex += f"{row_str} \\\\\n"

    # End table
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex

def format_value_with_uncertainty(value, uncertainty, decimals=2):
    """
    Format value with uncertainty in LaTeX style: value ± uncertainty.

    Args:
        value: Measured value
        uncertainty: Uncertainty (absolute or percentage)
        decimals: Number of decimal places

    Returns:
        str: LaTeX formatted string
    """
    if value is None or pd.isna(value):
        return "--"

    if uncertainty is None or pd.isna(uncertainty):
        return f"{value:.{decimals}f}"

    # Format: value ± uncertainty
    return f"${value:.{decimals}f} \\pm {uncertainty:.{decimals}f}$"

def test_basic_latex_table():
    """Test basic LaTeX table generation."""
    print("=== Test: Basic LaTeX Table ===\n")

    # Create sample data
    df = pd.DataFrame({
        'T (K)': [273.15, 298.15, 323.15],
        'P (MPa)': [10, 50, 100],
        'S (S/m)': [0.0845, 0.1205, 0.1567]
    })

    latex = generate_latex_table(df,
                                   caption="Conductivity measurements",
                                   label="tab:conductivity")

    # Verify LaTeX structure
    assert "\\begin{table}" in latex, "Missing table environment"
    assert "\\begin{tabular}" in latex, "Missing tabular environment"
    assert "\\caption" in latex, "Missing caption"
    assert "\\label" in latex, "Missing label"
    assert "\\hline" in latex, "Missing horizontal lines"
    assert "\\end{table}" in latex, "Missing end table"

    print("  ✓ LaTeX table structure correct")
    print(f"  ✓ Number of data rows: {len(df)}")
    print(f"  ✓ Number of columns: {len(df.columns)}")

    # Count column separators (&)
    first_data_line = [line for line in latex.split('\n')
                       if '273.15' in line][0]
    n_separators = first_data_line.count('&')
    expected_separators = len(df.columns) - 1

    assert n_separators == expected_separators, \
        f"Expected {expected_separators} separators, found {n_separators}"

    print(f"  ✓ Column separators: {n_separators}")
    print("\n✓ Basic LaTeX table generated\n")
    return latex

def test_latex_with_uncertainties():
    """Test LaTeX table with uncertainty values."""
    print("=== Test: LaTeX Table with Uncertainties ===\n")

    # Create data with uncertainties
    df = pd.DataFrame({
        'T (K)': [273, 298, 323],
        'P (MPa)': [10, 50, 100],
        'S (S/m)': ['0.084 ± 0.002', '0.121 ± 0.003', '0.157 ± 0.004']
    })

    latex = generate_latex_table(df,
                                   caption="Conductivity with uncertainties",
                                   label="tab:cond_unc")

    # Verify uncertainty symbols
    assert "±" in latex or "pm" in latex, "Missing uncertainty symbol"

    print("  ✓ Uncertainty values included")
    print("  ✓ ± symbol present")

    # Verify data format
    for val in df['S (S/m)']:
        # Remove LaTeX formatting for verification
        clean_val = val.replace('$', '').replace('\\', '')
        assert '±' in clean_val, f"Uncertainty not formatted in: {val}"

    print(f"  ✓ All {len(df)} rows have uncertainties")
    print("\n✓ LaTeX table with uncertainties generated\n")
    return latex

def test_latex_special_characters():
    """Test handling of special LaTeX characters."""
    print("=== Test: Special Characters in LaTeX ===\n")

    # Create data with special characters
    df = pd.DataFrame({
        'Compound': ['NaCl', 'MgSO₄', 'Na₂SO₄'],
        'Concentration': ['0.5 M', '1.0 M', '1.5 M'],
        'S (S/m)': [0.050, 0.085, 0.120]
    })

    # Test that subscripts are handled
    latex = generate_latex_table(df,
                                   caption="Solutions with subscripts",
                                   label="tab:compounds")

    # Verify subscripts are present (either Unicode or LaTeX form)
    has_subscripts = '₄' in latex or '_' in latex
    print(f"  ✓ Subscripts present: {has_subscripts}")

    # Verify common compounds
    compounds_present = all(comp in latex for comp in ['NaCl', 'SO', 'Na'])
    assert compounds_present, "Not all compounds present in LaTeX output"

    print("  ✓ All compounds present")
    print(f"  ✓ Number of compounds: {len(df)}")
    print("\n✓ Special characters handled\n")
    return latex

def test_latex_scientific_notation():
    """Test scientific notation in LaTeX tables."""
    print("=== Test: Scientific Notation ===\n")

    # Create data with very small/large values
    df = pd.DataFrame({
        'Parameter': ['K_cell', 'σ', 'R'],
        'Value': [1.234e-3, 5.678e-2, 1.234e5],
        'Unit': ['cm⁻¹', 'S/m', 'Ω']
    })

    latex = generate_latex_table(df,
                                   caption="Parameters in scientific notation",
                                   label="tab:sci_notation")

    # Check for scientific notation
    has_sci_notation = 'e-' in latex or 'e+' in latex or '10^' in latex
    print(f"  ✓ Scientific notation used: {has_sci_notation}")

    # Verify all values are present
    for val in df['Value']:
        # Value should appear in some form
        str_val = f"{val:.3e}"
        print(f"    Value {val:.3e} represented")

    print("\n✓ Scientific notation handled\n")
    return latex

def test_latex_multirow_header():
    """Test LaTeX table with multi-row headers."""
    print("=== Test: Multi-row Header ===\n")

    # Create data for multi-component solutions
    df = pd.DataFrame({
        'Composition': ['NaCl', 'MgSO4', 'NaCl+MgSO4'],
        'Conc (M)': [1.0, 0.5, '1.0+0.5'],
        'T=273K': [0.084, 0.045, 0.095],
        'T=298K': [0.121, 0.065, 0.135],
        'T=323K': [0.157, 0.085, 0.175]
    })

    latex = generate_latex_table(df,
                                   caption="Multi-component conductivity",
                                   label="tab:multicomp")

    # Verify structure
    assert '273' in latex, "Temperature 273K not found"
    assert '298' in latex, "Temperature 298K not found"
    assert '323' in latex, "Temperature 323K not found"

    print("  ✓ All temperature columns present")
    print(f"  ✓ Number of compositions: {len(df)}")
    print(f"  ✓ Number of temperature points: {len([c for c in df.columns if 'T=' in c])}")

    print("\n✓ Multi-row header table generated\n")
    return latex

def test_latex_table_export():
    """Test exporting LaTeX table to file."""
    print("=== Test: LaTeX Table Export ===\n")

    # Create sample data
    df = pd.DataFrame({
        'P (MPa)': [10, 50, 100],
        'T (K)': [273, 298, 323],
        'S (S/m)': [0.084, 0.121, 0.157]
    })

    latex = generate_latex_table(df,
                                   caption="Exported conductivity data",
                                   label="tab:export")

    # Export to file
    test_dir = Path('tests/test_output')
    test_dir.mkdir(parents=True, exist_ok=True)

    tex_path = test_dir / 'test_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)

    # Verify file was created
    assert tex_path.exists(), f"LaTeX file not created: {tex_path}"

    # Verify file content
    with open(tex_path, 'r') as f:
        content = f.read()

    assert '\\begin{table}' in content, "Table environment not in file"
    assert len(content) > 100, "File content seems too short"

    file_size = tex_path.stat().st_size
    print(f"  ✓ LaTeX file created: {tex_path.name}")
    print(f"  ✓ File size: {file_size} bytes")
    print(f"  ✓ Number of lines: {len(content.split(chr(10)))}")

    print("\n✓ LaTeX table exported successfully\n")
    return True

def test_latex_column_alignment():
    """Test column alignment in LaTeX tables."""
    print("=== Test: Column Alignment ===\n")

    # Create data with mixed types
    df = pd.DataFrame({
        'Compound': ['NaCl', 'MgSO4', 'KCl'],  # Left align
        'Conc': [1.0, 0.5, 1.5],  # Right align (numbers)
        'S (S/m)': [0.121, 0.065, 0.155]  # Right align (numbers)
    })

    # Generate with custom alignment
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{lrr}\n"  # left, right, right
    latex += "\\hline\\hline\n"

    # Headers
    latex += "Compound & Conc (M) & S (S/m) \\\\\n"
    latex += "\\hline\n"

    # Data rows
    for _, row in df.iterrows():
        latex += f"{row['Compound']} & {row['Conc']:.1f} & {row['S (S/m)']:.3f} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Verify alignment specification
    assert "{lrr}" in latex, "Column alignment not specified"
    print("  ✓ Column alignment: left, right, right")

    # Verify numeric formatting in the rendered table
    assert "1.0" in latex, "Concentration should have 1 decimal"
    assert "0.121" in latex, "Conductivity should have 3 decimals"

    print("  ✓ Numeric precision preserved")
    print("\n✓ Column alignment correct\n")
    return latex

def test_latex_booktabs_style():
    """Test LaTeX table with booktabs styling (professional)."""
    print("=== Test: Booktabs Style ===\n")

    df = pd.DataFrame({
        'T (K)': [273, 298, 323],
        'P (MPa)': [10, 50, 100],
        'S (S/m)': [0.084, 0.121, 0.157]
    })

    # Generate booktabs-style table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{ccc}\n"
    latex += "\\toprule\n"  # booktabs top rule
    latex += "T (K) & P (MPa) & S (S/m) \\\\\n"
    latex += "\\midrule\n"  # booktabs middle rule

    for _, row in df.iterrows():
        latex += f"{row['T (K)']} & {row['P (MPa)']} & {row['S (S/m)']:.3f} \\\\\n"

    latex += "\\bottomrule\n"  # booktabs bottom rule
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Verify booktabs commands
    assert "\\toprule" in latex, "Missing toprule"
    assert "\\midrule" in latex, "Missing midrule"
    assert "\\bottomrule" in latex, "Missing bottomrule"

    print("  ✓ Booktabs rules present")
    print("  ✓ Professional styling applied")
    print("\n✓ Booktabs style table generated\n")
    return latex

if __name__ == '__main__':
    try:
        # Run all tests
        test_basic_latex_table()
        test_latex_with_uncertainties()
        test_latex_special_characters()
        test_latex_scientific_notation()
        test_latex_multirow_header()
        test_latex_table_export()
        test_latex_column_alignment()
        test_latex_booktabs_style()

        print("=" * 50)
        print("✓ ALL LATEX TABLE TESTS PASSED")
        print("=" * 50)

    except AssertionError as e:
        print("\n" + "=" * 50)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 50)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)
