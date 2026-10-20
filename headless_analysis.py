#!/usr/bin/env python
"""
Headless analysis mode for HiPOZ.

Uses the same calibration logic as the GUI but runs programmatically
when a CSV/JSON config file specifies standards and measurements.
"""

import logging
import numpy as np
from pathlib import Path
import pandas as pd

from analysis_config import bath_state

log = logging.getLogger('HiPOZ')


def _state(sol, entry):
    """(P, T) for a result row: chiller-bath config values (see
    bath_state), else the Gamry file header, else the config entry."""
    bath = bath_state(entry, getattr(sol, 'P_MPa', None))
    if bath:
        return bath
    P = sol.P_MPa if getattr(sol, 'P_MPa', None) is not None else entry.get('P_MPa', '')
    T = sol.T_K if getattr(sol, 'T_K', None) is not None else entry.get('T_K', '')
    return P, T

def build_solution_lookup(measurements):
    """
    Build a {basename: Solution} lookup from an iterable of Solution objects.

    Args:
        measurements: Iterable of Solution objects (e.g. one folder's meas array)

    Returns:
        Dict keyed by file basename.
    """
    lookup = {}
    for solution in measurements:
        if solution is not None and getattr(solution, 'file', None):
            lookup[Path(solution.file).name] = solution
    return lookup


def run_headless_analysis(timeseries, analysis_config, output_dir=None,
                          solution_lookup=None):
    """
    Run calibration and measurement analysis without GUI.

    Uses the same logic as DataSelector but programmatically:
    1. For each calibration group:
       - Compute cell constant from standards: K = sigma_std * R
       - Apply K to measurements: sigma = K / R
    2. Save results to CSV in same directory as input config

    Args:
        timeseries: TimeSeries object with organized data
        analysis_config: AnalysisConfig object
        output_dir: Optional output directory. If None, saves to config file's directory.
        solution_lookup: Optional {basename: Solution} lookup. If None, one is
            built from all solutions across timeseries (all folders combined).
            Pass a per-folder lookup to keep results scoped to a single folder.

    Returns:
        DataFrame with analyzed results
    """
    log.info("Running headless analysis mode")

    # Build lookup of solutions by filename (all folders) unless one was provided
    if solution_lookup is None:
        solution_lookup = {}
        for date_data in timeseries.allMeas:
            if date_data is not None:
                for solution in date_data:
                    if solution is not None:
                        fname = Path(solution.file).name
                        solution_lookup[fname] = solution

    log.info(f"Found {len(solution_lookup)} solution objects")

    results = []

    # Process each calibration group
    for group_idx, group in enumerate(analysis_config.calibration_groups):
        group_name = group['name']
        log.info(f"\nProcessing {group_name}")
        log.info(f"  Standards: {len(group['standards'])}")
        log.info(f"  Measurements: {len(group['measurements'])}")

        # === Step 1: Compute cell constant from standards ===
        K_cell_values = []
        K_cell_errors = []

        for std_entry in group['standards']:
            if isinstance(std_entry, dict):
                std_fname = std_entry['filename']
                sigma_std = std_entry.get('conductivity_Sm')

                if not sigma_std:
                    log.warning(f"  Standard {std_fname} missing conductivity_Sm, skipping")
                    continue

                # Find the solution object
                if std_fname not in solution_lookup:
                    log.warning(f"  Standard {std_fname} not found in data, skipping")
                    continue

                sol = solution_lookup[std_fname]

                if not hasattr(sol, 'Rcalc_ohm') or sol.Rcalc_ohm is None:
                    log.warning(f"  Standard {std_fname} has no Rcalc_ohm (circuit fit failed?), skipping")
                    continue

                # Compute cell constant: K = sigma * R
                K_cell = sigma_std * sol.Rcalc_ohm
                K_cell_err = sigma_std * sol.Runc_ohm if hasattr(sol, 'Runc_ohm') and sol.Runc_ohm else 0

                K_cell_values.append(K_cell)
                K_cell_errors.append(K_cell_err)

                log.info(f"  {std_fname}: σ={sigma_std:.2e} S/m, R={sol.Rcalc_ohm:.4e} Ω → K={K_cell:.4e} S/m·Ω")

        if not K_cell_values:
            log.error(f"  No valid standards found for {group_name}, cannot compute K_cell")
            continue

        # Average cell constant across standards
        K_cell_mean = np.mean(K_cell_values)
        K_cell_std = np.std(K_cell_values)
        K_cell_unc = np.sqrt(K_cell_std**2 + np.mean([e**2 for e in K_cell_errors]))

        log.info(f"  Cell constant: K = {K_cell_mean:.4e} ± {K_cell_unc:.4e} S/m·Ω")
        log.info(f"  (from {len(K_cell_values)} standard(s))")

        # === Step 2: Apply cell constant to measurements ===
        for meas_entry in group['measurements']:
            if isinstance(meas_entry, dict):
                meas_fname = meas_entry['filename']

                # Check if excluded
                if meas_entry.get('exclude'):
                    log.info(f"  Skipping excluded measurement: {meas_fname}")
                    continue

                # Find the solution object
                if meas_fname not in solution_lookup:
                    log.warning(f"  Measurement {meas_fname} not found in data, skipping")
                    continue

                sol = solution_lookup[meas_fname]

                if not hasattr(sol, 'Rcalc_ohm') or sol.Rcalc_ohm is None:
                    log.warning(f"  Measurement {meas_fname} has no Rcalc_ohm (circuit fit failed?), skipping")
                    continue

                # Compute conductivity: sigma = K / R
                sigma_calc = K_cell_mean / sol.Rcalc_ohm
                R_unc = sol.Runc_ohm if hasattr(sol, 'Runc_ohm') and sol.Runc_ohm else 0
                sigma_calc_unc = sigma_calc * np.sqrt(
                    (K_cell_unc / K_cell_mean)**2 +
                    (R_unc / sol.Rcalc_ohm)**2
                ) if R_unc > 0 else 0

                P_val, T_val = _state(sol, meas_entry)

                # Build result row
                result = {
                    'group_name': group_name,
                    'filename': meas_fname,
                    'P_MPa': P_val,
                    'T_K': T_val,
                    'type': 'measurement',
                    'comp': meas_entry.get('comp', ''),
                    'w_ppt': meas_entry.get('w_ppt', ''),
                    'w_molal': meas_entry.get('w_molal', ''),
                    'R_ohm': sol.Rcalc_ohm,
                    'R_ohm_unc': R_unc,
                    'K_cell': K_cell_mean,
                    'K_cell_unc': K_cell_unc,
                    'conductivity_Sm': sigma_calc,
                    'conductivity_Sm_unc': sigma_calc_unc,
                    'exclude': meas_entry.get('exclude', ''),
                    'notes': meas_entry.get('notes', '')
                }

                results.append(result)

                log.info(f"  {meas_fname}: R={sol.Rcalc_ohm:.4e} Ω → σ={sigma_calc:.4e} ± {sigma_calc_unc:.4e} S/m")

        # Also add standards to results for completeness
        for std_entry in group['standards']:
            if isinstance(std_entry, dict):
                std_fname = std_entry['filename']

                if std_fname not in solution_lookup:
                    continue

                sol = solution_lookup[std_fname]

                P_val, T_val = _state(sol, std_entry)

                result = {
                    'group_name': group_name,
                    'filename': std_fname,
                    'P_MPa': P_val,
                    'T_K': T_val,
                    'type': 'standard',
                    'comp': std_entry.get('comp', ''),
                    'w_ppt': std_entry.get('w_ppt', ''),
                    'w_molal': std_entry.get('w_molal', ''),
                    'R_ohm': getattr(sol, 'Rcalc_ohm', np.nan),
                    'R_ohm_unc': getattr(sol, 'Runc_ohm', 0),
                    'K_cell': K_cell_mean,
                    'K_cell_unc': K_cell_unc,
                    'conductivity_Sm': std_entry.get('conductivity_Sm', ''),
                    'conductivity_Sm_unc': '',
                    'exclude': std_entry.get('exclude', ''),
                    'notes': std_entry.get('notes', '')
                }
                results.append(result)

    if not results:
        log.error("No results generated!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Determine output directory
    if output_dir is None:
        # Save in same directory as config file
        if analysis_config and analysis_config.config_path:
            output_path = Path(analysis_config.config_path).parent
        else:
            output_path = Path('hipoz_exports')
            log.warning("No config path found, falling back to hipoz_exports/")
    else:
        output_path = Path(output_dir)

    output_path.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save results CSV in same directory as config
    results_file = output_path / f'hipoz_{timestamp}_results.csv'
    df.to_csv(results_file, index=False)
    log.info(f"\nResults saved to: {results_file}")
    log.info(f"Total measurements analyzed: {len([r for r in results if r['type'] == 'measurement'])}")
    log.info(f"Total standards: {len([r for r in results if r['type'] == 'standard'])}")

    # Update the config file with computed conductivities
    _update_config_with_conductivities(analysis_config, results, output_path)

    # Also create _analyzed versions for backup (harmonize CSV ↔ JSON)
    _save_config_with_results(analysis_config, output_path, timestamp)

    return df


def _update_config_with_conductivities(analysis_config, results, output_path: Path):
    """
    Update the original config files (CSV and JSON) with computed conductivity values.
    This writes back to the original config file, not the _analyzed versions.

    Args:
        analysis_config: AnalysisConfig object with original config path
        results: List of result dictionaries from run_headless_analysis
        output_path: Directory where config files are located
    """
    if not analysis_config or not analysis_config.config_path:
        log.warning("No config path found, cannot update with conductivities")
        return

    # Build lookup of computed conductivities by filename
    conductivity_lookup = {}
    for result in results:
        if result['type'] == 'measurement' and result.get('conductivity_Sm'):
            filename = result['filename']
            conductivity_lookup[filename] = result['conductivity_Sm']

    if not conductivity_lookup:
        log.debug("No computed conductivities to write back")
        return

    log.info(f"Updating config files with {len(conductivity_lookup)} computed conductivities")

    # Update the calibration_groups structure in memory
    updated_count = 0
    for group in analysis_config.calibration_groups:
        for meas in group.get('measurements', []):
            if isinstance(meas, dict):
                filename = meas.get('filename')
                if filename in conductivity_lookup:
                    meas['conductivity_Sm'] = conductivity_lookup[filename]
                    updated_count += 1

    log.info(f"Updated {updated_count} measurement entries with computed conductivities")

    # Merge into the file itself, not calibration_groups: the loaded view
    # lacks excluded rows and extra columns, which must survive.
    from config_store import update_config
    config_path = Path(analysis_config.config_path)
    updates = {f: {'conductivity_Sm': s} for f, s in conductivity_lookup.items()}
    update_config(config_path, updates, add_missing=False)
    log.info(f"Writing updated config to CSV: {config_path.with_suffix('.csv')}")
    log.info(f"Writing updated config to JSON: {config_path.with_suffix('.json')}")
    log.info("✓ Original config files updated with computed conductivities")


def _save_config_with_results(analysis_config, output_dir: Path, timestamp: str):
    """
    Save a backup of the config with analysis results as
    <config>_analyzed.csv and <config>_analyzed.json.
    """
    from config_store import read_rows, write_rows

    if not analysis_config or not analysis_config.config_path:
        return

    config_path = Path(analysis_config.config_path)
    fieldnames, rows = read_rows(config_path)
    csv_output = Path(output_dir) / f"{config_path.stem}_analyzed.csv"
    log.info(f"Updating CSV config: {csv_output}")
    log.info(f"Creating matching JSON: {csv_output.with_suffix('.json')}")
    write_rows(csv_output, rows, fieldnames,
               json_extra={'generated_by': 'HiPOZ headless analysis',
                           'timestamp': timestamp, 'source': str(config_path)})


def should_run_headless(analysis_config):
    """
    Determine if we can run in headless mode.

    Requirements:
    - Config file exists
    - At least one group has standards with conductivity values
    - At least one group has measurements

    Returns:
        bool: True if headless mode is possible
    """
    if not analysis_config:
        return False

    for group in analysis_config.calibration_groups:
        # Check for standards with conductivity
        has_valid_standards = False
        for std in group['standards']:
            if isinstance(std, dict) and std.get('conductivity_Sm'):
                has_valid_standards = True
                break

        # Check for measurements
        has_measurements = len(group['measurements']) > 0

        if has_valid_standards and has_measurements:
            return True

    return False
