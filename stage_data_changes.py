import argparse
import subprocess
import time
from pathlib import Path


EXCLUDED_DATA_FOLDERS = {"20260928", "20261001"}


def stage_data_changes(repo_root: Path) -> bool:
    data_result = subprocess.run(
        ["git", "add", "-A", "--", "data"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    conductivity_folders = []
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for folder in data_dir.rglob("*"):
            if folder.is_dir() and "conductivity" in folder.name.casefold():
                relative_folder = folder.relative_to(data_dir)
                if relative_folder.parts and relative_folder.parts[0] in EXCLUDED_DATA_FOLDERS:
                    continue
                conductivity_folders.append(str(folder.relative_to(repo_root)))

    conductivity_result = subprocess.run(
        ["git", "add", "-f", "--", *conductivity_folders],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    ) if conductivity_folders else data_result

    analysis_result = subprocess.run(
        ["git", "add", "-f", "--", "data/*/zAnalysis*.csv", "data/*/zAnalysis*.json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        data_result.returncode == 0
        and conductivity_result.returncode == 0
        and analysis_result.returncode == 0
    )


def watch(repo_root: Path, interval_seconds: float) -> None:
    while True:
        stage_data_changes(repo_root)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    watch(Path(__file__).resolve().parent, args.interval)
