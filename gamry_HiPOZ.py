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

from analysis_config import AnalysisConfig, generate_config_from_directory
from headless_analysis import run_headless_analysis, should_run_headless, build_solution_lookup
from calibration_sharing import parse_cal_map, borrow_calibration

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
        '--cal-map',
        type=str,
        nargs='+',
        default=None,
        metavar='TARGET=SOURCE',
        help='Explicitly borrow calibration standards from another day. '
             'Format TARGET=SOURCE using folder names, e.g. '
             '--cal-map 20250819Cortes=20250818Cortes 20250820Cortes=20250818Cortes. '
             'Standard files are physically copied into TARGET and recorded in its '
             'zAnalysis config. Re-running with a different SOURCE re-associates '
             '(previous borrowed standards are removed first). No association is '
             'assumed unless specified here.'
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
    Prompt user to select one or more data directories using a GUI dialog.

    The native directory dialog only allows a single selection, so a
    non-native QFileDialog with extended selection is used to permit picking
    multiple folders at once.

    Returns:
        List of directory names (not full paths, just the folder names)
    """
    from PyQt5.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView
    import os

    # Start in data/ directory if it exists
    start_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(start_dir):
        start_dir = os.getcwd()

    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    dialog.setWindowTitle("Select Data Directory(ies) - use Ctrl/Cmd or Shift for multiple")
    dialog.setDirectory(start_dir)

    # Enable multi-selection on the internal views (non-native dialog only)
    for view in dialog.findChildren(QListView):
        if view.objectName() == 'listView':
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    for view in dialog.findChildren(QTreeView):
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)

    if dialog.exec_():
        selected_paths = dialog.selectedFiles()
        # Extract just the directory names (not full paths)
        dir_names = [os.path.basename(path) for path in selected_paths]
        log.info(f"User selected directories: {dir_names}")
        return dir_names
    else:
        log.info("User cancelled directory selection")
        return []

def find_config_in_dir(data_dir):
    """
    Look for a calibration/analysis config file inside a single data directory.
    Prioritizes CSV (Excel-friendly) over JSON, and matches any zAnalysis*.csv
    so folder-name suffixes (e.g. '20250818Cortes' vs 'zAnalysis20250818.csv')
    are handled.

    Args:
        data_dir: Data directory name (under data/)

    Returns:
        Path to config file if found, None otherwise
    """
    data_path = os.path.join('data', data_dir)
    if not os.path.exists(data_path):
        return None

    # Prefer any zAnalysis*.csv, then zAnalysis*.json
    for pattern in ('zAnalysis*.csv', 'zAnalysis*.json'):
        matches = sorted(glob(os.path.join(data_path, pattern)))
        if matches:
            log.info(f"Found analysis config: {matches[0]}")
            return matches[0]

    # Backward compatibility: old naming conventions
    legacy_names = ['calibration_config.csv', 'calibration.csv',
                    'calibration_config.json', 'calibration.json']
    for config_name in legacy_names:
        config_path = os.path.join(data_path, config_name)
        if os.path.exists(config_path):
            log.info(f"Found calibration config in data directory: {config_path}")
            return config_path

    return None


def build_folder_config_map(data_dirs):
    """
    Build a {folder: config_path} map for the given folders.

    If a folder has no config but contains measurement data, a template
    config is auto-generated (zAnalysis<folder>.csv) by scanning the folder.

    Args:
        data_dirs: List of data directory names (under data/)

    Returns:
        Dict {folder_name: config_path}. Folders with no config and no data
        are omitted.
    """
    config_map = {}
    for data_dir in data_dirs:
        data_path = os.path.join('data', data_dir)
        if not os.path.exists(data_path):
            log.error(f"Data directory does not exist: {data_path}")
            continue

        config_path = find_config_in_dir(data_dir)

        if not config_path:
            # Auto-generate a template config if the folder has data files
            has_data = bool(glob(os.path.join(data_path, 'Conductivity*', '*.txt')))
            if has_data:
                config_path = os.path.join(data_path, f'zAnalysis{data_dir}.csv')
                log.info(f"No config in {data_dir}; auto-generating {config_path}")
                generate_config_from_directory(data_path, output_path=config_path,
                                               format='csv')
            else:
                log.warning(f"No config and no data found for {data_dir}")
                continue

        config_map[data_dir] = config_path

    return config_map

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

    def _harmonize(cpath):
        """Best-effort CSV<->JSON harmonization for a config path."""
        try:
            from harmonize_config import harmonize_config
            harmonize_config(Path(cpath), force=True, verbose=False, auto_detect=True)
            log.debug(f"Auto-harmonized config files for {cpath}")
        except Exception as e:
            log.warning(f"Could not auto-harmonize {cpath}: {e}")

    # Load calibration config.
    # Legacy mode: a single --config applies to all data (one combined output).
    # Default mode: each folder has its own config, output stays in that folder.
    legacy_single = bool(args.config)
    analysis_config = None          # used in legacy mode / GUI fallback
    analysis_configs = {}           # {folder: AnalysisConfig} in per-folder mode

    if legacy_single:
        config_path = args.config
        log.info(f"Loading calibration configuration from: {config_path}")
        analysis_config = AnalysisConfig(config_path)
        _harmonize(config_path)
    else:
        # Discover (or auto-generate) a config per folder
        config_map = build_folder_config_map(dates_to_process)

        for date, cpath in config_map.items():
            _harmonize(cpath)
            analysis_configs[date] = AnalysisConfig(cpath)

        # Apply explicit calibration borrowing (--cal-map TARGET=SOURCE)
        cal_map = parse_cal_map(args.cal_map)
        for target, source in cal_map.items():
            if target not in analysis_configs:
                log.error(f"cal-map target '{target}' is not among processed folders; skipping")
                continue

            source_cpath = config_map.get(source) or find_config_in_dir(source)
            if not source_cpath:
                log.error(f"cal-map source '{source}' has no config/standards; skipping {target}")
                continue

            source_config = AnalysisConfig(source_cpath)
            ok = borrow_calibration(target, source,
                                    analysis_configs[target], source_config,
                                    config_map[target])
            if ok:
                # Reload target config from disk to reflect written borrowed entries
                analysis_configs[target] = AnalysisConfig(config_map[target])

        if not analysis_configs:
            log.info("No calibration configs found. Will use manual GUI calibration.")

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

    # === Legacy single-config mode: one combined analysis, one output ===
    if legacy_single:
        use_headless = False
        if args.headless:
            if not analysis_config or not should_run_headless(analysis_config):
                log.error("--headless flag requires a complete config file with standards and measurements")
                sys.exit(1)
            use_headless = True
        elif not args.gui:
            use_headless = analysis_config and should_run_headless(analysis_config)

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

    # === Per-folder mode: analyze each folder, output into its own folder ===
    else:
        ran_any_headless = False
        gui_config = None

        for d_ind, date in enumerate(dates_to_process):
            ac = analysis_configs.get(date)
            meas = all_meas[d_ind] if d_ind < len(all_meas) else None

            if ac is None or meas is None or len(meas) == 0:
                if ac is not None:
                    gui_config = gui_config or ac
                continue

            eligible = should_run_headless(ac)

            if args.gui or not (args.headless or eligible):
                if args.headless and not eligible:
                    log.warning(f"{date}: no complete standards/measurements for headless; deferring to GUI")
                gui_config = gui_config or ac
                continue

            out_dir = os.path.join('data', date)
            lookup = build_solution_lookup(meas)
            log.info("\n" + "="*70)
            log.info(f"HEADLESS ANALYSIS: {date}  ->  {out_dir}")
            log.info("="*70 + "\n")
            try:
                results_df = run_headless_analysis(timeseries, ac,
                                                   output_dir=out_dir,
                                                   solution_lookup=lookup)
                if results_df is not None:
                    ran_any_headless = True
                else:
                    log.error(f"{date}: headless analysis returned no results")
                    gui_config = gui_config or ac
            except Exception as e:
                log.error(f"{date}: headless analysis failed: {str(e)}")
                gui_config = gui_config or ac

        if ran_any_headless and gui_config is None and not args.gui:
            log.info("\n" + "="*70)
            log.info("ANALYSIS COMPLETE (per-folder)")
            log.info("="*70)
            log.info("Results saved into each folder. Use --gui to visualize or adjust.")
            sys.exit(0)

        # Choose a config for GUI fallback
        analysis_config = gui_config
        if analysis_config is None and analysis_configs:
            analysis_config = next(iter(analysis_configs.values()))

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
