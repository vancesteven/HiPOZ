#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to add date prefix to Gamry data files that don't have it.

Usage:
    python add_date_to_filenames.py [data_directory] [date_string]

Example:
    python add_date_to_filenames.py data/20250815Cortes 20250815

This script examines all .txt files in the specified directory (recursively)
and renames files that don't have the date prefix by adding it.
"""

import os
import sys
import shutil
from glob import glob
from datetime import datetime

def add_date_to_files(directory, date_string):
    """
    Add date prefix to all Gamry data files in directory that don't have it.

    Args:
        directory (str): Path to directory containing Gamry data files
        date_string (str): Date string to add (e.g., "20250813")

    Returns:
        int: Number of files renamed
    """
    # Validate date string format (should be YYYYMMDD)
    try:
        datetime.strptime(date_string, "%Y%m%d")
    except ValueError:
        print(f"Error: Date string '{date_string}' is not in the correct format (YYYYMMDD)")
        return 0

    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return 0

    # Find all .txt files recursively
    txt_files = glob(os.path.join(directory, "**/*.txt"), recursive=True)

    # Counter for renamed files
    renamed_count = 0

    for file_path in txt_files:
        # Get directory and filename
        dir_path, filename = os.path.split(file_path)

        # Skip if filename already starts with date string
        if filename.startswith(date_string):
            continue

        # Check if filename has typical Gamry timestamp format at the beginning (HHMMSS_)
        time_prefix = filename[:7]  # e.g., "110153_"
        if len(time_prefix) == 7 and time_prefix[:6].isdigit() and time_prefix[6] == '_':
            # Add date to filename
            new_filename = f"{date_string}_{filename}"
            new_path = os.path.join(dir_path, new_filename)

            # Rename the file
            print(f"Renaming: {filename} -> {new_filename}")
            shutil.move(file_path, new_path)
            renamed_count += 1

    return renamed_count

def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python add_date_to_filenames.py [data_directory] [date_string]")
        print("Example: python add_date_to_filenames.py data/20250815Cortes 20250815")
        return

    directory = sys.argv[1]
    date_string = sys.argv[2]

    # Confirm with user
    print(f"This will add the date prefix '{date_string}' to all Gamry data files")
    print(f"in directory '{directory}' that don't already have it.")
    confirm = input("Proceed? (y/n): ")

    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    # Add date to filenames
    renamed_count = add_date_to_files(directory, date_string)

    print(f"\nDone! {renamed_count} files were renamed.")

if __name__ == "__main__":
    main()