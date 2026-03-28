#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamry file utilities for HiPOZ - handles file operations for Gamry data files.

This module provides functions for adding date prefixes to Gamry data files,
ensuring consistent naming conventions across datasets.
"""

import os
import shutil
from glob import glob
from datetime import datetime
import logging
import re

# Set up logging
log = logging.getLogger('HiPOZ')

def add_date_to_filenames(directory, date_string, dry_run=False, recursive=True, subdirs_only=False):
    """
    Add date prefix to all Gamry data files in directory that don't have it.

    Args:
        directory (str): Path to directory containing Gamry data files
        date_string (str): Date string to add (e.g., "20250813")
        dry_run (bool): If True, just print what would be done without actually renaming files
        recursive (bool): If True, search subdirectories recursively
        subdirs_only (bool): If True, only process subdirectories (not files in main directory)

    Returns:
        tuple: (renamed_files, error_files) - Lists of renamed files and any that had errors
    """
    # Validate date string format (should be YYYYMMDD)
    try:
        datetime.strptime(date_string, "%Y%m%d")
    except ValueError:
        log.error(f"Date string '{date_string}' is not in the correct format (YYYYMMDD)")
        return [], [f"Invalid date format: {date_string}"]

    # Ensure directory exists
    if not os.path.isdir(directory):
        log.error(f"Directory '{directory}' does not exist")
        return [], [f"Directory not found: {directory}"]

    # Find all .txt files
    pattern = os.path.join(directory, "**/*.txt") if recursive else os.path.join(directory, "*.txt")

    # Directories to process
    dirs_to_process = []

    if subdirs_only:
        # Get all subdirectories and process them separately
        subdirs = [d for d in glob(os.path.join(directory, "*"), recursive=False) if os.path.isdir(d)]
        for subdir in subdirs:
            subpattern = os.path.join(subdir, "*.txt")
            dirs_to_process.append((subdir, glob(subpattern, recursive=False)))
    else:
        # Process all files in the main directory and subdirectories if recursive is True
        dirs_to_process.append((directory, glob(pattern, recursive=recursive)))

    # Lists to track renamed files and errors
    renamed_files = []
    error_files = []

    for current_dir, txt_files in dirs_to_process:
        for file_path in txt_files:
            # Get directory and filename
            dir_path, filename = os.path.split(file_path)

            # Skip if filename already starts with date string
            if filename.startswith(date_string):
                continue

            # Check if filename has typical Gamry timestamp format at the beginning (HHMMSS_)
            if re.match(r'^\d{6}_', filename):
                # Add date to filename
                new_filename = f"{date_string}_{filename}"
                new_path = os.path.join(dir_path, new_filename)

                # Rename the file
                action = "Would rename" if dry_run else "Renaming"
                log.info(f"{action}: {filename} -> {new_filename}")

                if not dry_run:
                    try:
                        shutil.move(file_path, new_path)
                        renamed_files.append((file_path, new_path))
                    except Exception as e:
                        log.error(f"Error renaming {file_path}: {str(e)}")
                        error_files.append((file_path, str(e)))

    return renamed_files, error_files

def extract_date_from_directory(directory_path):
    """
    Extract date from a directory name (assuming format contains YYYYMMDD).

    Args:
        directory_path (str): Path to directory

    Returns:
        str: Date string if found, None otherwise
    """
    # Extract just the directory name from the path
    dir_name = os.path.basename(os.path.normpath(directory_path))

    # Look for YYYYMMDD pattern in the directory name
    date_match = re.search(r'(\d{8})', dir_name)

    if date_match:
        date_str = date_match.group(1)
        # Validate it's a real date
        try:
            datetime.strptime(date_str, "%Y%m%d")
            return date_str
        except ValueError:
            pass

    return None

def auto_process_directory(directory_path, dry_run=False):
    """
    Automatically process a directory, extracting the date from the directory name.

    Args:
        directory_path (str): Path to directory
        dry_run (bool): If True, just print what would be done without actually renaming files

    Returns:
        tuple: (date_string, renamed_count, error_count) - Date used and counts of renamed/error files
    """
    # Try to extract date from directory name
    date_string = extract_date_from_directory(directory_path)

    if not date_string:
        log.error(f"Could not extract date from directory name: {directory_path}")
        return None, 0, 0

    # Find the ConductivityData_Default subdirectory
    conductivity_dir = os.path.join(directory_path, "ConductivityData_Default")
    target_dir = conductivity_dir if os.path.isdir(conductivity_dir) else directory_path

    # Add date to filenames
    renamed_files, error_files = add_date_to_filenames(target_dir, date_string, dry_run=dry_run)

    return date_string, len(renamed_files), len(error_files)

if __name__ == "__main__":
    # Example usage when run directly
    import sys

    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python gamry_file_utils.py [data_directory] [date_string (optional)]")
        print("Example: python gamry_file_utils.py data/20250815Cortes 20250815")
        sys.exit(1)

    directory = sys.argv[1]

    if len(sys.argv) >= 3:
        # Use provided date string
        date_string = sys.argv[2]
        renamed, errors = add_date_to_filenames(directory, date_string)
        print(f"Done! {len(renamed)} files renamed, {len(errors)} errors.")
    else:
        # Try to automatically extract date from directory name
        date_string, renamed_count, error_count = auto_process_directory(directory)
        if date_string:
            print(f"Used date: {date_string}")
            print(f"Done! {renamed_count} files renamed, {error_count} errors.")
        else:
            print("Could not determine date from directory name. Please provide date as second argument.")