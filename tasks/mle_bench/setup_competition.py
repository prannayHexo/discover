#!/usr/bin/env python3
"""
Generic setup script for any mle-bench Kaggle competition.

Usage:
  python3 setup_competition.py -c ventilator-pressure-prediction --data-dir /path/to/csvs
  python3 setup_competition.py -c spaceship-titanic --data-dir /path/to/data
  python3 setup_competition.py --list
"""

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MLEBENCH_REPO = Path(__file__).resolve().parent.parent.parent.parent / "mle-bench"
MLEBENCH_GIT_URL = "https://github.com/openai/mle-bench.git"


def ensure_mlebench_installed(repo_path: Path):
    """Make sure mlebench is importable, installing it if needed."""
    if _can_import("mlebench"):
        return

    repo_path = repo_path.resolve()
    if not (repo_path / "setup.py").exists() and not (repo_path / "pyproject.toml").exists():
        print(f"Cloning mle-bench into {repo_path}...")
        subprocess.check_call(["git", "clone", MLEBENCH_GIT_URL, str(repo_path)])

    print(f"Installing mle-bench from {repo_path}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_path)])

    # Force re-import
    importlib.invalidate_caches()
    if not _can_import("mlebench"):
        print("ERROR: mlebench still not importable after install.", file=sys.stderr)
        sys.exit(1)


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def list_competitions():
    from mlebench.registry import registry
    ids = registry.list_competition_ids()
    print(f"Available competitions ({len(ids)}):")
    for cid in ids:
        print(f"  {cid}")


def setup_competition(competition_id: str, data_dir: Path, skip_checksums: bool):
    from mlebench.registry import registry

    # Validate competition exists
    all_ids = registry.list_competition_ids()
    if competition_id not in all_ids:
        print(f"ERROR: '{competition_id}' not found in registry.", file=sys.stderr)
        print(f"Use --list to see available competitions.", file=sys.stderr)
        sys.exit(1)

    comp = registry.get_competition(competition_id)

    # 1. Copy raw data
    print(f"Setting up raw data for '{competition_id}'...")
    comp.raw_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"ERROR: --data-dir '{data_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    files_copied = 0
    for src in data_dir.iterdir():
        dst = comp.raw_dir / src.name
        if src.is_file():
            shutil.copy2(src, dst)
            files_copied += 1
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            files_copied += 1

    print(f"  Copied {files_copied} items to {comp.raw_dir}")

    # 2. Prepare public/private splits
    comp.public_dir.mkdir(parents=True, exist_ok=True)
    comp.private_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running prepare function...")
    comp.prepare_fn(raw=comp.raw_dir, public=comp.public_dir, private=comp.private_dir)

    # 3. Write description
    with open(comp.public_dir / "description.md", "w") as f:
        f.write(comp.description)

    # 4. Verify
    public_files = list(comp.public_dir.iterdir())
    private_files = list(comp.private_dir.iterdir())
    print(f"  Public dir:  {len(public_files)} files")
    print(f"  Private dir: {len(private_files)} files")

    if not public_files:
        print("WARNING: Public directory is empty!", file=sys.stderr)
    if not private_files:
        print("WARNING: Private directory is empty!", file=sys.stderr)

    print(f"\nSetup complete for '{competition_id}'")
    print(f"  Public data:  {comp.public_dir}")
    print(f"  Private data: {comp.private_dir}")
    print(f"\nRun with:")
    print(f"  python3 search.py --env mle_bench --problem_idx {competition_id} --rounds 5 --samples 4")


def main():
    parser = argparse.ArgumentParser(description="Set up any mle-bench competition for search.py")
    parser.add_argument("-c", "--competition", help="Competition ID (e.g. ventilator-pressure-prediction)")
    parser.add_argument("--data-dir", type=Path, help="Directory containing raw data files (CSVs etc.)")
    parser.add_argument("--mlebench-repo", type=Path, default=DEFAULT_MLEBENCH_REPO,
                        help="Path to mle-bench repo (default: ../mle-bench relative to project root)")
    parser.add_argument("--list", action="store_true", help="List all available competition IDs")
    parser.add_argument("--skip-checksums", action="store_true",
                        help="Skip checksum verification (use when providing your own data)")
    args = parser.parse_args()

    ensure_mlebench_installed(args.mlebench_repo)

    if args.list:
        list_competitions()
        return

    if not args.competition:
        parser.error("--competition / -c is required (or use --list)")
    if not args.data_dir:
        parser.error("--data-dir is required")

    setup_competition(args.competition, args.data_dir.resolve(), args.skip_checksums)


if __name__ == "__main__":
    main()
