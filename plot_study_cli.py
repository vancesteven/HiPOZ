#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized CLI for plotting conductivity study data.

This tool provides a unified interface for generating publication-quality
conductivity plots from benchtop and Gamry impedance data for any study.

Usage:
    # Plot a study with default settings
    python plot_study_cli.py mahboub2026

    # Plot with custom configuration
    python plot_study_cli.py cortes2026 --compounds NaCl KCl --show-delta

    # List available studies
    python plot_study_cli.py --list

    # Create configuration for new study
    python plot_study_cli.py --init newstudy

Features:
    - Automatic data loading from standardized CSV format
    - Gamry impedance overlay integration
    - McCleskey model comparison
    - Publication-ready PDF outputs
    - Configurable styling and formats
    - Batch processing of multiple compounds
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from study_plots import (
    load_study_data,
    plot_study_concentration,
    plot_study_temperature
)
from gamry_integration import (
    load_gamry_results,
    extract_compound_overlay
)
from config_plots import (
    FONTSIZE_AXIS_LABEL,
    FONTSIZE_TITLE,
    FONTSIZE_LEGEND,
    COLORMAP_CONCENTRATION,
    COLORMAP_TEMPERATURE
)


# ========================================
# Study Registry
# ========================================

STUDIES = {
    'mahboub2026': {
        'name': 'Mahboub et al. (2026)',
        'benchtop_csv': 'mahboub2026/Mahboub2026BenchtopData.csv',
        'gamry_dirs': ['data/20250813Mahboub2026', 'data/20250815Mahboub2026'],
        'output_dir': 'mahboub2026/mahboub_plots',
        'compounds': ['NaCl', 'MgSO4', 'NH4Cl', 'Na2CO3', 'Mixture'],
        'latex_names': {
            'MgSO4': r'MgSO$_4$',
            'NH4Cl': r'NH$_4$Cl',
            'Na2CO3': r'Na$_2$CO$_3$',
            'Mixture': r'MgSO$_4$:NaCl:Na$_2$CO$_3$ (1:1:1)'
        },
        'show_delta': {
            'NaCl': True,
            'MgSO4': True,
            'NH4Cl': True,
            'Na2CO3': True,
            'Mixture': False
        }
    },
    'cortes2026': {
        'name': 'Cortes et al. (2026)',
        'benchtop_csv': 'cortes2026/Cortes2026BenchtopData.csv',
        'gamry_dirs': ['data/20250813Cortes', 'data/20250814Cortes', 'data/20250815Cortes'],
        'output_dir': 'cortes2026/cortes_plots',
        'compounds': 'auto',  # Auto-detect from data
        'latex_names': {
            'MgSO4': r'MgSO$_4$',
            'Na2SO4': r'Na$_2$SO$_4$',
            'Na2CO3': r'Na$_2$CO$_3$',
            'NaCl:MgSO4_1:1': r'NaCl:MgSO$_4$ (1:1)',
            'NaCl:MgSO4_2:1': r'NaCl:MgSO$_4$ (2:1)',
            'NaCl:MgSO4_1:2': r'NaCl:MgSO$_4$ (1:2)',
            'Na2SO4:KCl_1:1': r'Na$_2$SO$_4$:KCl (1:1)',
            'Na2SO4:KCl_2:1': r'Na$_2$SO$_4$:KCl (2:1)',
            'Na2SO4:KCl_1:2': r'Na$_2$SO$_4$:KCl (1:2)',
        },
        'show_delta': 'auto'  # Show for single salts, not mixtures/organics
    }
}


# ========================================
# Functions
# ========================================

def list_studies():
    """Print available studies."""
    print("=" * 70)
    print("Available Studies")
    print("=" * 70)
    print()

    for study_id, config in STUDIES.items():
        print(f"  {study_id}")
        print(f"    Name: {config['name']}")
        print(f"    Data: {config['benchtop_csv']}")
        print(f"    Gamry: {len(config['gamry_dirs'])} dataset(s)")
        print()


def init_study_config(study_id):
    """Create configuration template for new study."""
    config_file = f"{study_id}_config.json"

    template = {
        "name": f"{study_id} Study",
        "benchtop_csv": f"{study_id}/{study_id.capitalize()}BenchtopData.csv",
        "gamry_dirs": [f"data/{study_id}_dataset1"],
        "output_dir": f"{study_id}/{study_id}_plots",
        "compounds": "auto",
        "latex_names": {
            "NaCl": "NaCl",
            "MgSO4": r"MgSO$_4$"
        },
        "show_delta": "auto"
    }

    with open(config_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Created configuration template: {config_file}")
    print()
    print("Next steps:")
    print(f"  1. Edit {config_file} with your study details")
    print(f"  2. Create benchtop data CSV: {template['benchtop_csv']}")
    print(f"  3. Run HiPOZ analysis on Gamry data")
    print(f"  4. Run: python plot_study_cli.py {study_id}")
    print()


def should_show_delta(compound, config):
    """Determine if McCleskey delta should be shown for this compound."""
    show_delta = config.get('show_delta', 'auto')

    if isinstance(show_delta, dict):
        return show_delta.get(compound, False)
    elif show_delta == 'auto':
        # Auto: Show for simple salts, not mixtures or organics
        simple_salts = ['KCl', 'NaCl', 'MgSO4', 'Na2SO4', 'NH4Cl', 'Na2CO3']
        return compound in simple_salts
    elif isinstance(show_delta, bool):
        return show_delta
    else:
        return False


def get_latex_name(compound, config):
    """Get LaTeX-formatted name for compound."""
    latex_names = config.get('latex_names', {})
    return latex_names.get(compound, compound)


def plot_study(study_id, config, compound_filter=None, format='pdf', dpi=300):
    """
    Generate all plots for a study.

    Parameters
    ----------
    study_id : str
        Study identifier
    config : dict
        Study configuration
    compound_filter : list, optional
        List of compounds to plot (default: all)
    format : str
        Output format ('pdf', 'png', 'both')
    dpi : int
        Resolution for raster outputs
    """
    print("=" * 70)
    print(f"{config['name']} - Conductivity Plots")
    print("=" * 70)
    print()

    # Load benchtop data
    benchtop_file = config['benchtop_csv']
    if not os.path.exists(benchtop_file):
        print(f"ERROR: Benchtop data not found: {benchtop_file}")
        print()
        print("Please create benchtop data CSV first.")
        return

    print(f"Loading benchtop data from {benchtop_file}...")
    benchtop_data = load_study_data(benchtop_file)
    print(f"  Found {len(benchtop_data)} compounds")
    print()

    # Load Gamry data from all specified directories
    gamry_dfs = []
    for data_dir in config['gamry_dirs']:
        df = load_gamry_results(data_dir, verbose=True)
        if df is not None:
            gamry_dfs.append(df)

    # Combine Gamry data
    if gamry_dfs:
        import pandas as pd
        gamry_df = pd.concat(gamry_dfs, ignore_index=True)
        print(f"Combined Gamry data: {len(gamry_df)} total measurements")
        print()
    else:
        gamry_df = None
        print("No Gamry data found (benchtop-only plots will be generated)")
        print()

    # Determine compounds to plot
    if config['compounds'] == 'auto':
        compounds_to_plot = list(benchtop_data.keys())
    else:
        compounds_to_plot = config['compounds']

    # Apply compound filter
    if compound_filter:
        compounds_to_plot = [c for c in compounds_to_plot if c in compound_filter]
        print(f"Filtering to compounds: {', '.join(compounds_to_plot)}")
        print()

    # Create output directory
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    plot_number = 1
    for compound in compounds_to_plot:
        latex_name = get_latex_name(compound, config)
        show_delta = should_show_delta(compound, config)

        # Extract Gamry overlay data
        gamry_overlay = extract_compound_overlay(gamry_df, compound) if gamry_df is not None else None

        # Generate output filenames
        safe_name = compound.lower().replace(':', '').replace(' ', '_')

        # Plot 1: σ vs Concentration
        print(f"Plot {plot_number}: {compound} σ vs Concentration...")

        for fmt in (['pdf', 'png'] if format == 'both' else [format]):
            output_file = os.path.join(output_dir, f'{safe_name}_vs_concentration.{fmt}')
            plot_study_concentration(
                data=benchtop_data,
                compound=compound,
                output_file=output_file,
                gamry_data=gamry_overlay,
                show_delta=show_delta,
                compound_latex=latex_name,
                colormap=COLORMAP_CONCENTRATION,
                fontsize_label=FONTSIZE_AXIS_LABEL,
                fontsize_title=FONTSIZE_TITLE,
                fontsize_legend=FONTSIZE_LEGEND,
                dpi=dpi if fmt == 'png' else None
            )
            print(f"  Saved: {output_file}")

        if gamry_overlay:
            print(f"  Gamry overlay: {len(gamry_overlay[0])} points")
        plot_number += 1
        print()

        # Plot 2: σ vs Temperature
        print(f"Plot {plot_number}: {compound} σ vs Temperature...")

        for fmt in (['pdf', 'png'] if format == 'both' else [format]):
            output_file = os.path.join(output_dir, f'{safe_name}_vs_temperature.{fmt}')
            plot_study_temperature(
                data=benchtop_data,
                compound=compound,
                output_file=output_file,
                show_delta=show_delta,
                compound_latex=latex_name,
                colormap=COLORMAP_TEMPERATURE,
                fontsize_label=FONTSIZE_AXIS_LABEL,
                fontsize_title=FONTSIZE_TITLE,
                fontsize_legend=FONTSIZE_LEGEND,
                dpi=dpi if fmt == 'png' else None
            )
            print(f"  Saved: {output_file}")

        plot_number += 1
        print()

    # Summary
    print("=" * 70)
    print(f"All plots saved to: {output_dir}/")
    print("=" * 70)
    print()
    print(f"Generated {len(compounds_to_plot)} × 2 = {len(compounds_to_plot) * 2} plots")
    print()


# ========================================
# Main
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality conductivity plots for any study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot Mahboub study
  python plot_study_cli.py mahboub2026

  # Plot Cortes study with specific compounds
  python plot_study_cli.py cortes2026 --compounds NaCl KCl

  # Generate PNG outputs
  python plot_study_cli.py mahboub2026 --format png --dpi 300

  # List available studies
  python plot_study_cli.py --list

  # Initialize new study configuration
  python plot_study_cli.py --init newstudy2026
        """
    )

    parser.add_argument('study', nargs='?', help='Study identifier (e.g., mahboub2026, cortes2026)')
    parser.add_argument('--list', '-l', action='store_true', help='List available studies')
    parser.add_argument('--init', metavar='STUDY_ID', help='Initialize new study configuration')
    parser.add_argument('--compounds', '-c', nargs='+', help='Specific compounds to plot')
    parser.add_argument('--format', '-f', choices=['pdf', 'png', 'both'], default='pdf',
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution for PNG output (default: 300)')
    parser.add_argument('--config', metavar='FILE', help='Load study configuration from JSON file')

    args = parser.parse_args()

    # Handle special commands
    if args.list:
        list_studies()
        return

    if args.init:
        init_study_config(args.init)
        return

    # Require study ID
    if not args.study and not args.config:
        parser.print_help()
        print()
        print("ERROR: Please specify a study ID or use --list to see available studies")
        sys.exit(1)

    # Load configuration
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        study_id = Path(args.config).stem
    elif args.study in STUDIES:
        study_id = args.study
        config = STUDIES[study_id]
    else:
        print(f"ERROR: Unknown study '{args.study}'")
        print()
        print("Available studies:")
        for sid in STUDIES.keys():
            print(f"  - {sid}")
        print()
        print("Or create a new study configuration:")
        print(f"  python plot_study_cli.py --init {args.study}")
        sys.exit(1)

    # Generate plots
    plot_study(study_id, config, args.compounds, args.format, args.dpi)


if __name__ == '__main__':
    main()
