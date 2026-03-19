import os, sys, re
import numpy as np
from glob import glob
import logging
from datetime import datetime as dtime
import argparse
from pathlib import Path

from gamryTools import Solution, CalStdFit, TimeSeries
from gamryPlots import plot_y, plot_z, plot_zvsf, plot_phasevsf, plot_zfit, plot_timeseries

from PyQt5.QtWidgets import QApplication
from hipoz_data_selector_gui import DataSelector

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis_config import AnalysisConfig
from headless_analysis import run_headless_analysis, should_run_headless

# import sys
# from PyQt5.QtWidgets import QApplication
# from FileSelector import FileSelector
#
# app = QApplication(sys.argv)  # Create an application if one doesn't exist
# ex = FileSelector()
# ex.show()
# app.exec_()  # Start the event loop

# After the window is closed, you can access the selected files
# calibration_files = ex.calibration_files
# data_files = ex.data_files

# Assign logger
log = logging.getLogger('HiPOZ')
stream = logging.StreamHandler(sys.stdout)
stream.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
log.setLevel(logging.DEBUG)
log.addHandler(stream)

pan_data = False

# Default dates - can be overridden by command line --dates or GUI directory selection
# Leave empty to prompt user for directory selection
dates = []  # Will prompt for directory if empty and no --dates specified
# dates = ['20221016']; # values read at 5e-4S/m
# dates = ['20221014']; # using the 1bar std yield around 5 S/m not 8. THere's a zero conductivity point at high pressure
# dates = ['20221010','20221011']
# dates = ['20220922','20220923']#,'20220924','20221010','20221011']
# dates = ['20220924']#,'20221010','20221011']
circ_type = 'CPE'  # Options are 'CPE', 'RC', and 'RC-R'. If desired, a circuit string can be entered here instead.
initial_guess = None  # Required when circ_type is not in the above list. Ignored otherwise.
cmap_name = 'viridis'
out_fig_name = 'GamryMeas'
fig_size = (6,6)

f_range_hz = 1e3*np.array([10,100])

xtn = 'png'
plot_air = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HiPOZ: High-Pressure Ocean world impedance analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GUI - will prompt for directory selection:
  python gamry_HiP.py

  # Specify directory(ies) from command line:
  python gamry_HiP.py --dates 20250815Mahboub

  # Process multiple directories:
  python gamry_HiP.py --dates 20250813 20250814 20250815

  # Headless mode with specific directory:
  python gamry_HiP.py --headless --dates 20250815Mahboub

  # Run with specific config file:
  python gamry_HiP.py --config data/20250815/zAnalysis20250815.csv

  # Harmonize CSV and JSON (after editing one):
  python gamry_HiP.py --harmonize data/20250815/zAnalysis20250815.csv
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to calibration configuration file (JSON or CSV) that specifies '
             'which files are standards and which are measurements'
    )

    parser.add_argument(
        '--dates',
        type=str,
        nargs='+',
        default=None,
        help='Specify data directory(ies) to process (e.g., --dates 20250815Mahboub). '
             'If not provided, GUI will prompt for directory selection.'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Force GUI mode even when config file exists (for manual adjustments)'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Force headless mode (skip GUI, requires complete config file)'
    )

    parser.add_argument(
        '--plot-svsp',
        action='store_true',
        help='Generate conductivity vs pressure (S vs P) plot in headless mode'
    )

    parser.add_argument(
        '--plot-bode',
        action='store_true',
        help='Generate Bode plots in headless mode'
    )

    parser.add_argument(
        '--plot-nyquist',
        action='store_true',
        help='Generate Nyquist plots in headless mode'
    )

    parser.add_argument(
        '--plot-all',
        action='store_true',
        help='Generate all available plots in headless mode'
    )

    parser.add_argument(
        '--plot-sigma-conc',
        action='store_true',
        help='Generate conductivity vs concentration plot in headless mode'
    )

    parser.add_argument(
        '--plot-sigma-temp',
        action='store_true',
        help='Generate conductivity vs temperature plot in headless mode'
    )

    parser.add_argument(
        '--harmonize',
        type=str,
        metavar='FILE',
        help='Harmonize CSV↔JSON config file (creates matching format) and exit'
    )

    return parser.parse_args()

def select_data_directories():
    """
    Prompt user to select data directories using GUI dialog.

    Returns:
        List of directory names (not full paths, just the folder names)
    """
    from PyQt5.QtWidgets import QFileDialog
    import os

    # Start in data/ directory if it exists
    start_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(start_dir):
        start_dir = os.getcwd()

    # Allow multiple directory selection
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    dialog.setWindowTitle("Select Data Directory(ies)")
    dialog.setDirectory(start_dir)

    if dialog.exec_():
        selected_paths = dialog.selectedFiles()
        # Extract just the directory names (not full paths)
        dir_names = [os.path.basename(path) for path in selected_paths]
        log.info(f"User selected directories: {dir_names}")
        return dir_names
    else:
        log.info("User cancelled directory selection")
        return []

def find_config_in_data_dirs(data_dirs):
    """
    Look for calibration config files in data directories.
    Prioritizes CSV (Excel-friendly) over JSON.

    Args:
        data_dirs: List of data directory names

    Returns:
        Path to config file if found, None otherwise
    """
    for data_dir in data_dirs:
        data_path = os.path.join('data', data_dir)
        if os.path.exists(data_path):
            # First priority: zAnalysis<date>.csv (Excel-friendly, new convention)
            analysis_csv = os.path.join(data_path, f'zAnalysis{data_dir}.csv')
            if os.path.exists(analysis_csv):
                log.info(f"Found CSV analysis config: {analysis_csv}")
                return analysis_csv

            # Second priority: zAnalysis<date>.json (new naming convention)
            analysis_json = os.path.join(data_path, f'zAnalysis{data_dir}.json')
            if os.path.exists(analysis_json):
                log.info(f"Found JSON analysis config: {analysis_json}")
                return analysis_json

            # Backward compatibility: old naming conventions
            legacy_names = ['calibration_config.csv', 'calibration.csv',
                           'calibration_config.json', 'calibration.json']
            for config_name in legacy_names:
                config_path = os.path.join(data_path, config_name)
                if os.path.exists(config_path):
                    log.info(f"Found calibration config in data directory: {config_path}")
                    return config_path

    return None

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Handle harmonize command (doesn't need GUI)
    if args.harmonize:
        from harmonize_config import harmonize_config
        result = harmonize_config(Path(args.harmonize), force=True, verbose=True, auto_detect=True)
        if result:
            log.info(f"✓ Config files harmonized successfully")
            sys.exit(0)
        else:
            log.error("Failed to harmonize config files")
            sys.exit(1)

    # Ensure QApplication is initialized early (needed for dialogs)
    app = QApplication(sys.argv)

    # Determine which directories to process
    dates_to_process = args.dates if args.dates else dates

    # If no directories specified, prompt user to select
    if not dates_to_process:
        log.info("No data directories specified. Prompting user for selection...")
        dates_to_process = select_data_directories()

        if not dates_to_process:
            log.error("No directories selected. Exiting.")
            sys.exit(0)

    log.info(f"Processing directories: {dates_to_process}")

    # Load calibration config
    analysis_config = None
    config_path = args.config

    # If no config specified on command line, look in data directories
    if not config_path:
        config_path = find_config_in_data_dirs(dates_to_process)

    if config_path:
        log.info(f"Loading calibration configuration from: {config_path}")
        analysis_config = AnalysisConfig(config_path)

        # Auto-harmonize: ensure CSV and JSON are in sync
        try:
            from harmonize_config import harmonize_config
            harmonize_config(Path(config_path), force=True, verbose=False, auto_detect=True)
            log.debug("Auto-harmonized config files (CSV ↔ JSON)")
        except Exception as e:
            log.warning(f"Could not auto-harmonize config files: {e}")
    else:
        log.info("No calibration config found. Will use manual GUI calibration.")
    # Import data
    add = None
    n_dates = np.size(dates_to_process)
    all_meas = np.empty(n_dates, dtype=object)
    for d_ind, this_date in enumerate(dates_to_process):
        log.info(f'Processing {this_date}/')
        path_head = os.path.join(os.path.join('data', this_date))

        if not os.path.exists(path_head):
            log.error(f"Data directory does not exist: {path_head}")
            continue

        gamry_files = glob(os.path.join(path_head, 'Conductivity*', '*.txt'))
        # gamry_files = [f for f in fList if re.search(this_date+'-'+'[0-9][0-9][0-9][0-9]_', f)]
        n_sweeps = np.size(gamry_files)

        if n_sweeps == 0:
            log.warning(f"No data files found in {path_head}/Conductivity*/*.txt")
            continue

        log.info(f"Found {n_sweeps} data files to process")
        meas = np.empty(n_sweeps, dtype=object)
        cal_std = CalStdFit(interpMethod='cubic')

        lf = len(gamry_files)
        successful_count = 0
        failed_files = []

        for i, file in enumerate(gamry_files):
            print(f'Processing measurement {i+1} of {lf}: {os.path.basename(file)}')
            try:
                meas[i] = Solution(cmap_name=cmap_name)
                meas[i].load_file(file, pan=pan_data)

                # if not np.isnan(meas[i].sigmaStd_Sm):
                #     meas[i].sigmaStdCalc_Sm = cal_std(meas[i].T_K, lbl_uScm=meas[i].lbl_uScm)
                # else:
                #     meas[i].sigmaStdCalc_Sm = 1e-8  # Default air conductivity

                meas[i].fit_circuit(circ_type=circ_type, initial_guess=initial_guess, print_circuit=False, basin_hopping=False, multiproc=True, f_range_hz=f_range_hz)
                successful_count += 1

            except Exception as e:
                log.error(f"Failed to process file {os.path.basename(file)}: {str(e)}")
                failed_files.append((file, str(e)))
                meas[i] = None  # Mark as failed

        # Filter out None entries (failed measurements)
        meas = np.array([m for m in meas if m is not None])
        log.info(f'Successfully processed {successful_count} of {lf} files for {this_date}')

        if failed_files:
            log.warning(f'Failed to process {len(failed_files)} file(s):')
            for failed_file, error in failed_files:
                log.warning(f'  - {os.path.basename(failed_file)}: {error}')

        all_meas[d_ind] = meas

    # Check if we have any valid measurements
    total_measurements = sum(len(m) for m in all_meas if m is not None and len(m) > 0)
    if total_measurements == 0:
        log.error("No valid measurements were successfully processed. Exiting.")
        sys.exit(1)

    log.info(f"Total valid measurements across all dates: {total_measurements}")

    try:
        timeseries = TimeSeries(all_meas)
        timeseries.organizeData()
        log.info('TimeSeries data organized successfully')
    except Exception as e:
        log.error(f"Failed to create or organize TimeSeries: {str(e)}")
        sys.exit(1)

    # Determine whether to use headless or GUI mode
    use_headless = False

    if args.headless:
        # User explicitly requested headless mode
        if not analysis_config or not should_run_headless(analysis_config):
            log.error("--headless flag requires a complete config file with standards and measurements")
            sys.exit(1)
        use_headless = True
    elif args.gui:
        # User explicitly requested GUI mode
        use_headless = False
    else:
        # Auto-detect: use headless if config is complete
        use_headless = analysis_config and should_run_headless(analysis_config)

    # Run headless mode if appropriate
    if use_headless:
        log.info("\n" + "="*70)
        log.info("HEADLESS ANALYSIS MODE")
        log.info("Config file has standards and measurements - running without GUI")
        log.info("="*70 + "\n")

        try:
            results_df = run_headless_analysis(timeseries, analysis_config)

            if results_df is not None:
                log.info("\n" + "="*70)
                log.info("ANALYSIS COMPLETE")
                log.info("="*70)
                log.info("Results have been saved. Use --gui flag to visualize or make adjustments.")
                sys.exit(0)
            else:
                log.error("Headless analysis failed, falling back to GUI mode")
        except Exception as e:
            log.error(f"Headless analysis failed: {str(e)}")
            log.info("Falling back to GUI mode...")

    # Fall back to GUI mode
    log.info("\n" + "="*70)
    log.info("GUI MODE")
    log.info("Launching interactive data selector")
    log.info("="*70 + "\n")

    try:
        ds = DataSelector(timeseries, analysis_config=analysis_config)
        ds.show()

        # Check if the main window is visible
        if not ds.isVisible():
            log.warning("The DataSelector window is not visible")

        # The geometry of the window can be printed to ensure it's within the visible area of your screen
        log.info(f"Window geometry: {ds.geometry()}")

    except Exception as e:
        log.error(f"Failed to launch DataSelector GUI: {str(e)}")
        sys.exit(1)

    # Start the event loop
    return_code = app.exec_()
    log.info(f"Event loop exited with code: {return_code}")
    sys.exit(return_code)

# plot_timeseries(all_meas, fig_size, out_fig_name, xtn)
if __name__ == "__main__":
    main()
# 1/Z_cell = 1/R + i*omega*C -- Pan et al. (2021): https://doi.org/10.1029/2021GL094020
