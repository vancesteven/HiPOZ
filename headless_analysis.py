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

log = logging.getLogger('HiPOZ')


def run_headless_analysis(timeseries, calib_config, output_dir=None):
    """
    Run calibration and measurement analysis without GUI.

    Uses the same logic as DataSelector but programmatically:
    1. For each calibration group:
       - Compute cell constant from standards: K = sigma_std * R
       - Apply K to measurements: sigma = K / R
    2. Save results to CSV in same directory as input config

    Args:
        timeseries: TimeSeries object with organized data
        calib_config: CalibrationConfig object
        output_dir: Optional output directory. If None, saves to config file's directory.

    Returns:
        DataFrame with analyzed results
    """
    log.info("Running headless analysis mode")

    # Build lookup of all solutions by filename
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
    for group_idx, group in enumerate(calib_config.calibration_groups):
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

                # Get P and T from solution object (highest priority) or config entry
                P_val = sol.P_MPa if hasattr(sol, 'P_MPa') and sol.P_MPa is not None else meas_entry.get('P_MPa', '')
                T_val = sol.T_K if hasattr(sol, 'T_K') and sol.T_K is not None else meas_entry.get('T_K', '')

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

                # Get P and T from solution object (highest priority) or config entry
                P_val = sol.P_MPa if hasattr(sol, 'P_MPa') and sol.P_MPa is not None else std_entry.get('P_MPa', '')
                T_val = sol.T_K if hasattr(sol, 'T_K') and sol.T_K is not None else std_entry.get('T_K', '')

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
        if calib_config and calib_config.config_path:
            output_path = Path(calib_config.config_path).parent
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
    _update_config_with_conductivities(calib_config, results, output_path)

    # Also create _analyzed versions for backup (harmonize CSV ↔ JSON)
    _save_config_with_results(calib_config, output_path, timestamp)

    return df


def _update_config_with_conductivities(calib_config, results, output_path: Path):
    """
    Update the original config files (CSV and JSON) with computed conductivity values.
    This writes back to the original config file, not the _analyzed versions.

    Args:
        calib_config: CalibrationConfig object with original config path
        results: List of result dictionaries from run_headless_analysis
        output_path: Directory where config files are located
    """
    if not calib_config or not calib_config.config_path:
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
    for group in calib_config.calibration_groups:
        for meas in group.get('measurements', []):
            if isinstance(meas, dict):
                filename = meas.get('filename')
                if filename in conductivity_lookup:
                    meas['conductivity_Sm'] = conductivity_lookup[filename]
                    updated_count += 1

    log.info(f"Updated {updated_count} measurement entries with computed conductivities")

    # Write back to both CSV and JSON formats
    config_path = Path(calib_config.config_path)
    is_csv_input = config_path.suffix.lower() == '.csv'

    import json
    import csv as csv_module

    # Update CSV
    csv_path = config_path if is_csv_input else config_path.with_suffix('.csv')
    log.info(f"Writing updated config to CSV: {csv_path}")

    fieldnames = ['group_name', 'filename', 'P_MPa', 'T_K', 'type',
                 'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes']

    with open(csv_path, 'w', newline='') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for group in calib_config.calibration_groups:
            group_name = group['name']

            # Write standards
            for std in group.get('standards', []):
                if isinstance(std, dict):
                    row = {
                        'group_name': group_name,
                        'filename': std.get('filename', ''),
                        'P_MPa': std.get('P_MPa', ''),
                        'T_K': std.get('T_K', ''),
                        'type': 'standard',
                        'conductivity_Sm': std.get('conductivity_Sm', ''),
                        'comp': std.get('comp', ''),
                        'w_ppt': std.get('w_ppt', ''),
                        'w_molal': std.get('w_molal', ''),
                        'exclude': 'x' if std.get('exclude') else '',
                        'notes': std.get('notes', '')
                    }
                    writer.writerow(row)

            # Write measurements (now with computed conductivities)
            for meas in group.get('measurements', []):
                if isinstance(meas, dict):
                    row = {
                        'group_name': group_name,
                        'filename': meas.get('filename', ''),
                        'P_MPa': meas.get('P_MPa', ''),
                        'T_K': meas.get('T_K', ''),
                        'type': 'measurement',
                        'conductivity_Sm': meas.get('conductivity_Sm', ''),  # Now populated!
                        'comp': meas.get('comp', ''),
                        'w_ppt': meas.get('w_ppt', ''),
                        'w_molal': meas.get('w_molal', ''),
                        'exclude': 'x' if meas.get('exclude') else '',
                        'notes': meas.get('notes', '')
                    }
                    writer.writerow(row)

    # Update JSON
    json_path = config_path if not is_csv_input else config_path.with_suffix('.json')
    log.info(f"Writing updated config to JSON: {json_path}")

    with open(json_path, 'w') as f:
        json.dump({
            'calibration_groups': calib_config.calibration_groups,
            'generated_by': 'HiPOZ headless analysis',
            'updated_with_conductivities': True
        }, f, indent=2)

    log.info("✓ Original config files updated with computed conductivities")


def _save_config_with_results(calib_config, output_dir: Path, timestamp: str):
    """
    Save/update config file with analysis results.
    If input was CSV, also create matching JSON. If input was JSON, also create matching CSV.
    """
    import json
    import csv as csv_module

    if not calib_config or not calib_config.config_path:
        return

    config_path = Path(calib_config.config_path)
    is_csv_input = config_path.suffix.lower() == '.csv'

    # Determine output paths for both formats
    base_name = config_path.stem  # e.g., "zAnalysis20250813Mahboub2026"
    csv_output = output_dir / f"{base_name}_analyzed.csv"
    json_output = output_dir / f"{base_name}_analyzed.json"

    if is_csv_input:
        # Input was CSV → also create/update matching JSON
        log.info(f"Updating CSV config: {csv_output}")
        # CSV is already up to date from original, just copy with timestamp
        import shutil
        shutil.copy(config_path, csv_output)

        # Create matching JSON
        log.info(f"Creating matching JSON: {json_output}")
        with open(json_output, 'w') as f:
            json.dump({
                'calibration_groups': calib_config.calibration_groups,
                'generated_by': 'HiPOZ headless analysis',
                'timestamp': timestamp,
                'source_csv': str(config_path)
            }, f, indent=2)

    else:
        # Input was JSON → also create/update matching CSV
        log.info(f"Updating JSON config: {json_output}")
        import shutil
        shutil.copy(config_path, json_output)

        # Create matching CSV
        log.info(f"Creating matching CSV: {csv_output}")
        with open(csv_output, 'w', newline='') as f:
            fieldnames = ['group_name', 'filename', 'P_MPa', 'T_K', 'type',
                         'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes']
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for group in calib_config.calibration_groups:
                group_name = group['name']

                # Write standards
                for std in group.get('standards', []):
                    if isinstance(std, dict):
                        row = {
                            'group_name': group_name,
                            'filename': std.get('filename', ''),
                            'P_MPa': std.get('P_MPa', ''),
                            'T_K': std.get('T_K', ''),
                            'type': 'standard',
                            'conductivity_Sm': std.get('conductivity_Sm', ''),
                            'comp': std.get('comp', ''),
                            'w_ppt': std.get('w_ppt', ''),
                            'w_molal': std.get('w_molal', ''),
                            'exclude': std.get('exclude', ''),
                            'notes': std.get('notes', '')
                        }
                        writer.writerow(row)

                # Write measurements
                for meas in group.get('measurements', []):
                    if isinstance(meas, dict):
                        row = {
                            'group_name': group_name,
                            'filename': meas.get('filename', ''),
                            'P_MPa': meas.get('P_MPa', ''),
                            'T_K': meas.get('T_K', ''),
                            'type': 'measurement',
                            'conductivity_Sm': '',
                            'comp': meas.get('comp', ''),
                            'w_ppt': meas.get('w_ppt', ''),
                            'w_molal': meas.get('w_molal', ''),
                            'exclude': meas.get('exclude', ''),
                            'notes': meas.get('notes', '')
                        }
                        writer.writerow(row)


def should_run_headless(calib_config):
    """
    Determine if we can run in headless mode.

    Requirements:
    - Config file exists
    - At least one group has standards with conductivity values
    - At least one group has measurements

    Returns:
        bool: True if headless mode is possible
    """
    if not calib_config:
        return False

    for group in calib_config.calibration_groups:
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
