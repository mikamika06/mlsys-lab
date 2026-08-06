import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_milestone_dir(milestone_name: str) -> Path:
    candidates = [
        Path(milestone_name),
        Path("milestones") / milestone_name,
        Path("tasks") / milestone_name,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"Could not find milestone directory for '{milestone_name}'")

def run_verification_target(milestone_dir: Path, target_name: str):
    target_dir = milestone_dir / target_name
    if not target_dir.exists():
        # Handle case where reference/skeleton might be single python files or inside target_dir
        if (milestone_dir / f"{target_name}.py").exists():
            target_dir = milestone_dir
        else:
            raise FileNotFoundError(f"Target '{target_name}' not found in {milestone_dir}")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{target_dir}:{milestone_dir}:{existing_pythonpath}".strip(":")

    test_file = milestone_dir / "tests.py"
    if not test_file.exists():
        test_files = list(milestone_dir.glob("test_*.py"))
        if test_files:
            test_file = test_files[0]

    if not test_file.exists():
        raise FileNotFoundError(f"No test file found in {milestone_dir}")

    cmd = [sys.executable, "-m", "pytest", str(test_file)] if import_pytest() else [sys.executable, str(test_file)]

    result = subprocess.run(cmd, cwd=milestone_dir, env=env, capture_output=True, text=True)
    return result

def import_pytest():
    try:
        import pytest
        return True
    except ImportError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify milestone reference and skeleton implementations.")
    parser.add_argument("milestone", help="Milestone directory or name")
    args = parser.parse_args()

    try:
        milestone_dir = find_milestone_dir(args.milestone)
    except FileNotFoundError as e:
        print(f"FAIL {args.milestone}: {e}")
        sys.exit(1)

    # 1. Test reference
    ref_res = run_verification_target(milestone_dir, "reference")
    if ref_res.returncode != 0:
        print(f"FAIL {args.milestone} reference failed execution:\n{ref_res.stderr or ref_res.stdout}")
        sys.exit(1)

    # 2. Test skeleton
    skel_res = run_verification_target(milestone_dir, "skeleton")
    if skel_res.returncode == 0:
        print(f"FAIL {args.milestone} skeleton passed tests, but skeleton must clear 0 tests (gate is not strict enough).")
        sys.exit(1)

    print(f"PASS {args.milestone}")

if __name__ == "__main__":
    main()
