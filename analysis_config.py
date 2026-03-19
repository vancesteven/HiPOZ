"""
Analysis configuration handler for HiPOZ.

Manages the complete analysis setup including standards (for calibration),
measurements, metadata (P, T, composition, concentrations), and organization
into calibration groups for bracketed measurements.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger('HiPOZ')


class AnalysisConfig:
    """
    Handles loading and applying analysis configuration from file.

    Manages the complete analysis setup including standards (for calibration),
    measurements, metadata (P, T, composition, concentrations), and organization
    into calibration groups for bracketed measurements.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize analysis config.

        Args:
            config_path: Path to configuration file (JSON or CSV). If None, no auto-config.
        """
        self.config_path = config_path
        self.calibration_groups = []  # List of calibration group dicts

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """
        Load configuration from JSON or CSV file.

        Args:
            config_path: Path to configuration file
        """
        config_file = Path(config_path)

        if not config_file.exists():
            log.error(f"Configuration file not found: {config_path}")
            return

        if config_file.suffix.lower() == '.json':
            self._load_json(config_file)
        elif config_file.suffix.lower() == '.csv':
            self._load_csv(config_file)
        else:
            log.error(f"Unsupported config format: {config_file.suffix}. Use .json or .csv")

    def _load_json(self, config_file: Path):
        """
        Load JSON configuration.

        Expected format:
        {
          "calibrations": [
            {
              "name": "Morning session",
              "standards": [
                "file1.txt",
                {"filename": "file2.txt", "conductivity_Sm": 0.0084, "comp": "KCl", "w_ppt": 1.0}
              ],
              "measurements": [
                "file3.txt",
                {"filename": "file4.txt", "comp": "NaCl", "w_ppt": 1000}
              ]
            }
          ]
        }

        Standards can be:
        - Simple string: filename (conductivity read from file metadata)
        - Dict: {"filename": "...", "conductivity_Sm": value, "comp": "KCl", "w_ppt": 1.0} (explicit values)

        Measurements can be:
        - Simple string: filename
        - Dict: {"filename": "...", "comp": "NaCl", "w_ppt": 1000} or {"filename": "...", "comp": "NaCl", "w_molal": 1.5}
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Support both 'calibrations' (old) and 'calibration_groups' (new) keys
            self.calibration_groups = config.get('calibration_groups', config.get('calibrations', []))

            log.info(f"Loaded {len(self.calibration_groups)} calibration group(s) from JSON")
            for i, group in enumerate(self.calibration_groups):
                name = group.get('name', f'Group {i+1}')
                n_std = len(group.get('standards', []))
                n_meas = len(group.get('measurements', []))
                log.info(f"  {name}: {n_std} standards, {n_meas} measurements")

        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            log.error(f"Error loading config: {e}")

    def _load_csv(self, config_file: Path):
        """
        Load CSV configuration.

        Expected columns: group_name, filename, P_MPa, T_K, type, conductivity_Sm, comp, w_ppt, w_molal, exclude
        (all optional except group_name, filename, type)
        Where type is 'standard' or 'measurement'
        Exclude can be: 'yes', 'true', '1', or 'x' to skip this file
        P_MPa and T_K are informational only (actual values loaded from data files)
        """
        try:
            groups_dict = {}

            with open(config_file, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    group_name = row.get('group_name', row.get('group', 'Default'))
                    filename = row.get('filename', '')
                    file_type = row.get('type', '').lower()
                    conductivity_str = row.get('conductivity_Sm', row.get('conductivity', ''))
                    comp = row.get('comp', row.get('composition', ''))
                    w_ppt_str = row.get('w_ppt', '')
                    w_molal_str = row.get('w_molal', '')
                    p_mpa_str = row.get('P_MPa', '')
                    t_k_str = row.get('T_K', '')
                    notes = row.get('notes', '')
                    exclude_str = row.get('exclude', '').lower().strip()

                    if not filename:
                        continue

                    # Skip if marked for exclusion (yes, true, 1, or x)
                    if exclude_str in ['yes', 'true', '1', 'x']:
                        log.info(f"Skipping excluded file: {filename}")
                        continue

                    if group_name not in groups_dict:
                        groups_dict[group_name] = {
                            'name': group_name,
                            'standards': [],
                            'measurements': []
                        }

                    if file_type == 'standard':
                        # Build dict entry with any specified values
                        std_entry = {'filename': filename}

                        if conductivity_str and conductivity_str.strip():
                            try:
                                std_entry['conductivity_Sm'] = float(conductivity_str)
                            except ValueError:
                                log.warning(f"Invalid conductivity value '{conductivity_str}' for {filename}")

                        if comp and comp.strip():
                            std_entry['comp'] = comp.strip()

                        if w_ppt_str and w_ppt_str.strip():
                            try:
                                std_entry['w_ppt'] = float(w_ppt_str)
                            except ValueError:
                                log.warning(f"Invalid w_ppt value '{w_ppt_str}' for {filename}")

                        if w_molal_str and w_molal_str.strip():
                            try:
                                std_entry['w_molal'] = float(w_molal_str)
                            except ValueError:
                                log.warning(f"Invalid w_molal value '{w_molal_str}' for {filename}")

                        if p_mpa_str and p_mpa_str.strip():
                            try:
                                std_entry['P_MPa'] = float(p_mpa_str)
                            except ValueError:
                                std_entry['P_MPa'] = p_mpa_str  # Keep as string if not numeric

                        if t_k_str and t_k_str.strip():
                            try:
                                std_entry['T_K'] = float(t_k_str)
                            except ValueError:
                                std_entry['T_K'] = t_k_str  # Keep as string if not numeric

                        if notes and notes.strip():
                            std_entry['notes'] = notes.strip()

                        # If only filename was provided, just use string
                        if len(std_entry) == 1:
                            std_entry = filename

                        groups_dict[group_name]['standards'].append(std_entry)

                    elif file_type == 'measurement':
                        # Build dict entry with any specified values
                        meas_entry = {'filename': filename}

                        if comp and comp.strip():
                            meas_entry['comp'] = comp.strip()

                        if w_ppt_str and w_ppt_str.strip():
                            try:
                                meas_entry['w_ppt'] = float(w_ppt_str)
                            except ValueError:
                                log.warning(f"Invalid w_ppt value '{w_ppt_str}' for {filename}")

                        if w_molal_str and w_molal_str.strip():
                            try:
                                meas_entry['w_molal'] = float(w_molal_str)
                            except ValueError:
                                log.warning(f"Invalid w_molal value '{w_molal_str}' for {filename}")

                        if p_mpa_str and p_mpa_str.strip():
                            try:
                                meas_entry['P_MPa'] = float(p_mpa_str)
                            except ValueError:
                                meas_entry['P_MPa'] = p_mpa_str  # Keep as string if not numeric

                        if t_k_str and t_k_str.strip():
                            try:
                                meas_entry['T_K'] = float(t_k_str)
                            except ValueError:
                                meas_entry['T_K'] = t_k_str  # Keep as string if not numeric

                        if notes and notes.strip():
                            meas_entry['notes'] = notes.strip()

                        # If only filename was provided, just use string
                        if len(meas_entry) == 1:
                            meas_entry = filename

                        groups_dict[group_name]['measurements'].append(meas_entry)

            self.calibration_groups = list(groups_dict.values())

            log.info(f"Loaded {len(self.calibration_groups)} calibration group(s) from CSV")
            for group in self.calibration_groups:
                log.info(f"  {group['name']}: {len(group['standards'])} standards, "
                        f"{len(group['measurements'])} measurements")

        except Exception as e:
            log.error(f"Error loading CSV config: {e}")

    def match_filename(self, search_name: str, target_name: str) -> bool:
        """
        Check if two filenames match (basename comparison).

        Args:
            search_name: Filename to search for (from config)
            target_name: Filename from timeseries (possibly full path)

        Returns:
            True if filenames match
        """
        search_base = Path(search_name).name
        target_base = Path(target_name).name
        return search_base == target_base

    def find_file_index(self, filename: str, timeseries) -> Optional[int]:
        """
        Find index of file in timeseries.

        Args:
            filename: Filename to find (can be basename or full path)
            timeseries: TimeSeries object

        Returns:
            Index if found, None otherwise
        """
        for i, ts_filename in enumerate(timeseries.filenames):
            if self.match_filename(filename, ts_filename):
                return i
        return None

    def apply_to_timeseries(self, timeseries) -> List[Dict]:
        """
        Apply all calibration groups to timeseries data.

        Args:
            timeseries: TimeSeries object with loaded measurement data

        Returns:
            List of result dicts, one per calibration group:
            {
              'name': str,
              'standard_indices': List[int],
              'measurement_indices': List[int],
              'cell_constant': float,
              'cell_constant_unc': float,
              'status': 'success' or 'error',
              'message': str
            }
        """
        results = []

        for group in self.calibration_groups:
            result = self._apply_calibration_group(group, timeseries)
            results.append(result)

        return results

    def _apply_calibration_group(self, group: Dict, timeseries) -> Dict:
        """Apply a single calibration group."""
        name = group.get('name', 'Unnamed')
        standard_files = group.get('standards', [])
        measurement_files = group.get('measurements', [])

        result = {
            'name': name,
            'standard_indices': [],
            'standard_conductivities': [],  # Store explicit conductivities if provided
            'standard_metadata': [],  # Store full metadata for each standard
            'measurement_indices': [],
            'cell_constant': 0.0,
            'cell_constant_unc': 0.0,
            'status': 'error',
            'message': ''
        }

        # Find standard indices
        # Standards can be either strings (filename) or dicts ({"filename": "...", "conductivity_Sm": value, "exclude": false})
        for std_entry in standard_files:
            if isinstance(std_entry, str):
                std_file = std_entry
                explicit_conductivity = None
                std_metadata = {
                    'conductivity_Sm': None,
                    'comp': None,
                    'w_ppt': None,
                    'w_molal': None
                }
                exclude = False
            elif isinstance(std_entry, dict):
                # Check if marked for exclusion
                exclude = std_entry.get('exclude', False)
                if exclude:
                    log.info(f"Skipping excluded standard: {std_entry.get('filename', 'unknown')}")
                    continue

                std_file = std_entry.get('filename', std_entry.get('file', ''))
                explicit_conductivity = std_entry.get('conductivity_Sm', std_entry.get('conductivity', None))
                std_metadata = {
                    'conductivity_Sm': explicit_conductivity,
                    'comp': std_entry.get('comp', std_entry.get('composition', None)),
                    'w_ppt': std_entry.get('w_ppt', None),
                    'w_molal': std_entry.get('w_molal', None)
                }
            else:
                log.warning(f"Invalid standard entry format: {std_entry}")
                continue

            idx = self.find_file_index(std_file, timeseries)
            if idx is not None:
                result['standard_indices'].append(idx)
                result['standard_conductivities'].append(explicit_conductivity)  # None if not specified
                result['standard_metadata'].append(std_metadata)  # Store full metadata
                log.debug(f"Found standard: {Path(std_file).name} at index {idx}")
                if explicit_conductivity is not None:
                    log.debug(f"  Using explicit conductivity: {explicit_conductivity} S/m")
            else:
                log.warning(f"Standard file not found: {std_file}")

        # Find measurement indices and store metadata
        result['measurement_metadata'] = []
        for meas_entry in measurement_files:
            if isinstance(meas_entry, str):
                meas_file = meas_entry
                metadata = {}
                exclude = False
            elif isinstance(meas_entry, dict):
                # Check if marked for exclusion
                exclude = meas_entry.get('exclude', False)
                if exclude:
                    log.info(f"Skipping excluded measurement: {meas_entry.get('filename', 'unknown')}")
                    continue

                meas_file = meas_entry.get('filename', meas_entry.get('file', ''))
                metadata = {
                    'comp': meas_entry.get('comp', meas_entry.get('composition', None)),
                    'w_ppt': meas_entry.get('w_ppt', None),
                    'w_molal': meas_entry.get('w_molal', None)
                }
            else:
                log.warning(f"Invalid measurement entry format: {meas_entry}")
                continue

            idx = self.find_file_index(meas_file, timeseries)
            if idx is not None:
                result['measurement_indices'].append(idx)
                result['measurement_metadata'].append(metadata)
                log.debug(f"Found measurement: {Path(meas_file).name} at index {idx}")
                if metadata.get('comp'):
                    log.debug(f"  Comp: {metadata.get('comp')}, w: {metadata.get('w_ppt') or metadata.get('w_molal')}")
            else:
                log.warning(f"Measurement file not found: {meas_file}")

        if not result['standard_indices']:
            result['message'] = f"No standards found for group '{name}'"
            log.warning(result['message'])
            return result

        # Calculate cell constant from standards
        cell_constants = []
        cell_const_uncertainties = []

        for i, idx in enumerate(result['standard_indices']):
            # Use explicit conductivity from config if provided, otherwise use value from file
            explicit_sigma = result['standard_conductivities'][i]
            if explicit_sigma is not None:
                sigma_Sm = explicit_sigma
                log.info(f"  Using explicit σ = {sigma_Sm} S/m from config")
            else:
                sigma_Sm = timeseries.conductivities_Sm[idx]
                if sigma_Sm is not None and sigma_Sm > 0:
                    log.info(f"  Using σ = {sigma_Sm} S/m from file metadata")

            R_ohm = timeseries.Rcalc_ohm[idx]
            dR_ohm = timeseries.uncertainties[idx]

            if sigma_Sm is not None and sigma_Sm > 0 and R_ohm > 0:
                k_cell = sigma_Sm * R_ohm
                dk_cell = dR_ohm  # Simplified uncertainty
                cell_constants.append(k_cell)
                cell_const_uncertainties.append(dk_cell)
                log.info(f"  Standard {idx}: σ={sigma_Sm:.4f} S/m, R={R_ohm:.2f} Ω, "
                        f"K_cell={k_cell:.2f} 1/m")

        if not cell_constants:
            result['message'] = f"No valid cell constants computed for group '{name}'"
            log.error(result['message'])
            return result

        # Compute mean cell constant
        n = len(cell_constants)
        result['cell_constant'] = float(np.mean(cell_constants))

        if n > 1:
            std_dev = np.std(cell_constants, ddof=1)
            mean_unc = np.mean(np.power(cell_const_uncertainties, 2))
            result['cell_constant_unc'] = np.sqrt(mean_unc / n + std_dev**2 / n)
        else:
            result['cell_constant_unc'] = float(cell_const_uncertainties[0])

        result['status'] = 'success'
        result['message'] = (f"K_cell = {result['cell_constant']:.2f} ± "
                           f"{result['cell_constant_unc']:.2f} 1/m from {n} standard(s)")

        log.info(f"Group '{name}': {result['message']}")

        return result


def create_example_config_json(output_path: str = "calibration_config_example.json"):
    """
    Create an example JSON configuration file with explicit filenames.

    Args:
        output_path: Where to save the example file
    """
    example_config = {
        "description": "HiPOZ Calibration Configuration - Multiple Bracketed Groups",
        "calibrations": [
            {
                "name": "Morning session (0800-1200)",
                "standards": [
                    {"filename": "Default_20250813_0800_KCl_84uScm_P100_T298.txt", "conductivity_Sm": 0.0084, "comp": "KCl", "w_ppt": 1.0},
                    {"filename": "Default_20250813_0805_KCl_447uScm_P100_T298.txt", "conductivity_Sm": 0.0447, "comp": "KCl", "w_ppt": 5.0},
                    "Default_20250813_0810_KCl_2070uScm_P100_T298.txt"
                ],
                "measurements": [
                    {"filename": "Default_20250813_0900_NaCl_1000ppt_P100_T298.txt", "comp": "NaCl", "w_ppt": 1000},
                    {"filename": "Default_20250813_0930_NaCl_1000ppt_P150_T298.txt", "comp": "NaCl", "w_ppt": 1000},
                    {"filename": "Default_20250813_1000_NaCl_1000ppt_P200_T298.txt", "comp": "NaCl", "w_ppt": 1000},
                    {"filename": "Default_20250813_1030_NaCl_1500ppt_P100_T298.txt", "comp": "NaCl", "w_ppt": 1500},
                    {"filename": "Default_20250813_1100_NaCl_1500ppt_P150_T298.txt", "comp": "NaCl", "w_ppt": 1500}
                ]
            },
            {
                "name": "Afternoon session (1200-1700)",
                "standards": [
                    {"filename": "Default_20250813_1200_KCl_84uScm_P100_T298.txt", "conductivity_Sm": 0.0084, "comp": "KCl", "w_ppt": 1.0},
                    {"filename": "Default_20250813_1205_KCl_447uScm_P100_T298.txt", "conductivity_Sm": 0.0447, "comp": "KCl", "w_ppt": 5.0}
                ],
                "measurements": [
                    {"filename": "Default_20250813_1300_NaCl_2000ppt_P100_T298.txt", "comp": "NaCl", "w_ppt": 2000},
                    {"filename": "Default_20250813_1330_NaCl_2000ppt_P150_T298.txt", "comp": "NaCl", "w_ppt": 2000},
                    {"filename": "Default_20250813_1400_MgSO4_500ppt_P100_T298.txt", "comp": "MgSO4", "w_ppt": 500}
                ]
            }
        ],
        "notes": [
            "Each calibration group has its own set of standards and measurements",
            "Standards can be:",
            "  - Simple string (filename): reads values from file metadata",
            "  - Dict with optional fields: conductivity_Sm, comp, w_ppt, w_molal, exclude",
            "Measurements can be:",
            "  - Simple string (filename): leaves comp and w blank in GUI",
            "  - Dict with optional fields: comp, w_ppt, w_molal, exclude",
            "To exclude files: add 'exclude': true to any entry",
            "  - Example: {\"filename\": \"bad_data.txt\", \"exclude\": true}",
            "Conductivity conversion: '84 uS/cm' = 0.0084 S/m, '447 uS/cm' = 0.0447 S/m",
            "w_ppt: concentration in g/kg solution, w_molal: mol/kg solvent",
            "Use this for bracketed measurements (standards before/after experiments)"
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)

    print(f"Example JSON configuration saved to: {output_path}")


def create_example_config_csv(output_path: str = "calibration_config_example.csv"):
    """
    Create an example CSV configuration file.

    Args:
        output_path: Where to save the example file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group_name', 'filename', 'P_MPa', 'T_K', 'type', 'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes'])

        # Morning group
        writer.writerow(['Morning', 'Default_20250813_0800_KCl_84uScm.txt', '', '', 'standard', '0.0084', 'KCl', '1.0', '', '', '84 uS/cm'])
        writer.writerow(['Morning', 'Default_20250813_0805_KCl_447uScm.txt', '', '', 'standard', '0.0447', 'KCl', '5.0', '', '', '447 uS/cm'])
        writer.writerow(['Morning', 'Default_20250813_0900_NaCl_1000ppt_P_100_T_298.txt', '100', '298', 'measurement', '', 'NaCl', '1000', '', '', ''])
        writer.writerow(['Morning', 'Default_20250813_0930_NaCl_1000ppt_P_150_T_298.txt', '150', '298', 'measurement', '', 'NaCl', '1000', '', '', ''])
        writer.writerow(['Morning', 'Default_20250813_0945_bad_data.txt', '', '', 'measurement', '', '', '', '', 'x', 'Excluded - bad data'])

        # Afternoon group
        writer.writerow(['Afternoon', 'Default_20250813_1200_KCl_84uScm.txt', '', '', 'standard', '0.0084', 'KCl', '1.0', '', '', ''])
        writer.writerow(['Afternoon', 'Default_20250813_1300_NaCl_2000ppt_P_100_T_298.txt', '100', '298', 'measurement', '', 'NaCl', '2000', '', '', ''])

    print(f"Example CSV configuration saved to: {output_path}")


def parse_pressure_temp_from_filename(filename: str) -> tuple:
    """
    Parse pressure (MPa) and temperature (K) from filename.

    Expected format: ...P_123_T_456.txt where P=123 MPa, T=456 K

    Args:
        filename: Filename to parse

    Returns:
        Tuple of (pressure, temperature) or ('', '') if not found
    """
    import re

    # Pattern: P_<number>_T_<number>
    match = re.search(r'P_(\d+)_T_(\d+)', filename)
    if match:
        p_val = match.group(1)
        t_val = match.group(2)
        return (p_val, t_val)

    # Return empty strings if not found
    return ('', '')


def generate_config_from_directory(data_dir: str, output_path: str = None, format: str = 'csv'):
    """
    Generate a template config by scanning a data directory.

    Args:
        data_dir: Path to data directory with ConductivityData_Default subdirectory
        output_path: Where to save the generated config (if None, uses zAnalysis<date>.csv or .json)
        format: Output format: 'csv' (default, Excel-friendly) or 'json'
    """
    from glob import glob

    # Auto-generate output path based on directory name and format if not specified
    if output_path is None:
        # Extract date from directory name (e.g., "data/20250813" -> "20250813")
        dir_name = Path(data_dir).name
        ext = '.csv' if format.lower() == 'csv' else '.json'
        output_path = f"zAnalysis{dir_name}{ext}"

    # Find all .txt files
    pattern = Path(data_dir) / "Conductivity*" / "*.txt"
    files = sorted(glob(str(pattern)))

    if not files:
        print(f"No .txt files found in {data_dir}/Conductivity*/")
        return

    # Separate KCl (standards) from others (measurements)
    standards = []
    measurements = []

    for f in files:
        basename = Path(f).name
        if 'KCl' in basename:
            standards.append(basename)
        else:
            measurements.append(basename)

    date_str = Path(data_dir).name

    if format.lower() == 'csv':
        # Generate CSV format (Excel-friendly)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header with P, T, and exclude columns
            writer.writerow(['group_name', 'filename', 'P_MPa', 'T_K', 'type', 'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes'])

            # Write standards
            for std in standards:
                # Parse P and T from filename (format: ...P_123_T_456.txt)
                p_val, t_val = parse_pressure_temp_from_filename(std)
                writer.writerow(['Group 1', std, p_val, t_val, 'standard', '', '', '', '', '', 'Add conductivity_Sm (e.g., 0.0084 for 84 uS/cm)'])

            # Write measurements
            for meas in measurements:
                # Parse P and T from filename
                p_val, t_val = parse_pressure_temp_from_filename(meas)
                writer.writerow(['Group 1', meas, p_val, t_val, 'measurement', '', '', '', '', '', 'Add comp (e.g., NaCl) and w_ppt (e.g., 1000)'])

        print(f"Generated CSV config with {len(standards)} standards and {len(measurements)} measurements")
        print(f"Saved to: {output_path}")
        print("\nOpen in Excel to edit:")
        print("  - Set conductivity_Sm for standards (e.g., 0.0084 for 84 µS/cm)")
        print("  - Set comp and w_ppt for measurements")
        print("  - Mark exclude='x' to skip files")
        print("  - Create multiple groups for bracketed measurements")
    else:
        # Generate JSON format
        config = {
            "description": f"Analysis for {date_str}",
            "date": date_str,
            "calibrations": [
                {
                    "name": "Calibration group 1",
                    "standards": [{"filename": f, "conductivity_Sm": "", "comp": "", "w_ppt": ""} for f in standards],
                    "measurements": [{"filename": f, "comp": "", "w_ppt": ""} for f in measurements]
                }
            ],
            "notes": [
                "Auto-generated - please review and edit as needed",
                "1. Add conductivity_Sm values to standards (e.g., 84 uS/cm = 0.0084 S/m)",
                "2. Add comp and w_ppt to measurements (e.g., 'NaCl', 1000)",
                "3. Add 'exclude': true to skip files",
                "4. Split into multiple groups if you measured standards at different times"
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Generated JSON config with {len(standards)} standards and {len(measurements)} measurements")
        print(f"Saved to: {output_path}")
        print("\nReview and edit the config file to:")
        print("  - Add conductivity_Sm, comp, w_ppt values")
        print("  - Add 'exclude': true to skip files")
        print("  - Split into multiple calibration groups if needed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        # Generate config from directory
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/20250813"

        # Check for --format flag
        format_type = 'csv'  # Default to CSV (Excel-friendly)
        output = None

        for i, arg in enumerate(sys.argv[3:], start=3):
            if arg in ['--format', '-f']:
                if i + 1 < len(sys.argv):
                    format_type = sys.argv[i + 1]
            elif arg not in ['json', 'csv'] or i == 0 or sys.argv[i-1] not in ['--format', '-f']:
                # This is the output path (not the format value)
                if '.' in arg:  # Has extension
                    output = arg
                    # Detect format from extension if not explicitly set
                    if arg.endswith('.json'):
                        format_type = 'json'
                    elif arg.endswith('.csv'):
                        format_type = 'csv'

        print(f"Scanning {data_dir}...")
        print(f"Output format: {format_type.upper()}")
        generate_config_from_directory(data_dir, output, format=format_type)
    else:
        # Generate example configs
        create_example_config_json()
        create_example_config_csv()
        print("\nExample configuration files created!")
        print("\nUsage:")
        print("  # Generate examples:")
        print("  python calibration_config.py")
        print()
        print("  # Generate CSV config (Excel-friendly, default):")
        print("  python calibration_config.py scan data/20250813")
        print("  python calibration_config.py scan data/20250813 --format csv")
        print()
        print("  # Generate JSON config:")
        print("  python calibration_config.py scan data/20250813 --format json")
        print()
        print("  # Custom output name (format auto-detected from extension):")
        print("  python calibration_config.py scan data/20250813 my_analysis.csv")
        print()
        print("  # Then run (auto-detects zAnalysis files in data dir):")
        print("  python gamry_HiPOZ.py")
        print()
        print("Notes:")
        print("  - CSV format is recommended for Excel users")
        print("  - Mark files with 'x' in exclude column to skip them")
        print("  - Create multiple groups for bracketed measurements")
