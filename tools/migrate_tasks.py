#!/usr/bin/env python3
"""One-shot migration of the task bank from the old repo into this one.

Copies every task directory, rewrites the imports that named the old package,
fixes the `sys.path` hacks that assumed a flat layout, and verifies a sample so
the migration cannot silently produce a bank that no longer grades.

    python3 tools/migrate_tasks.py --src ~/compression-arena          # dry run
    python3 tools/migrate_tasks.py --src ~/compression-arena --apply
    python3 tools/migrate_tasks.py --src ~/compression-arena --apply --verify 40

Safe to re-run: existing task directories are refreshed, so it can be used again
after the generator finishes more tasks.
"""
import argparse
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent

# The old package exposed everything flat; the new one groups the hardware models
# under `mlsys.sim` and keeps the task-facing helpers at the package root.
REWRITES = [
    ("from arena.cuda_sim import GPU", "from mlsys.sim import GPU"),
    ("from arena.cuda_sim import", "from mlsys.sim import"),
    ("from arena.cuda_c import CudaProgram", "from mlsys.sim import CudaProgram"),
    ("from arena.cuda_c import", "from mlsys.sim.cuda_c import"),
    ("from arena import cachesim", "from mlsys.sim import cache as cachesim"),
    ("from arena import cppabi", "from mlsys.sim import abi as cppabi"),
    ("from arena.cachesim import", "from mlsys.sim.cache import"),
    ("from arena.cppabi import", "from mlsys.sim.abi import"),
    ("from arena.scorers import", "from mlsys.scorers import"),
    ("from arena.probe import", "from mlsys.probe import"),
    ("from arena import", "from mlsys import"),
    ("import arena.cuda_sim", "import mlsys.sim.gpu"),
    ("import arena", "import mlsys"),
    # a handful of cuda tasks add the repo root to sys.path to reach the package;
    # under the src-layout that has to point at src/
    ('sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))',
     'sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))'),
]

SKIP_DIRS = {"__pycache__"}
SKIP_FILES = {".DS_Store", "solve.py"}   # solve.py is the learner's own attempt


def rewrite(text):
    for a, b in REWRITES:
        text = text.replace(a, b)
    return text


def copy_task(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for item in src_dir.iterdir():
        if item.name in SKIP_DIRS or item.name in SKIP_FILES:
            continue
        if item.is_dir():
            # A task may ship binary fixtures (fixtures/*.npy). Skipping every
            # directory silently produced tasks whose grader could not load its
            # own data, so subdirectories are mirrored, not dropped.
            seen.add(item.name)
            target_dir = dst_dir / item.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(item, target_dir,
                            ignore=shutil.ignore_patterns(*SKIP_DIRS, ".DS_Store"))
            continue
        seen.add(item.name)
        target = dst_dir / item.name
        if item.suffix == ".py":
            target.write_text(rewrite(item.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(item, target)
    # The upstream bank still ships native starters as solve.cpp / solve.cu, the
    # same file the learner edits. Here they become starter.cpp / starter.cu so an
    # attempt can never overwrite the shipped starter.
    meta = dst_dir / "meta.json"
    if meta.is_file():
        try:
            native = json.loads(meta.read_text()).get("native")
        except Exception:
            native = None
        ext = {"cpp": "cpp", "cuda": "cu"}.get(native)
        if ext and (dst_dir / f"solve.{ext}").is_file():
            (dst_dir / f"starter.{ext}").write_text(
                (dst_dir / f"solve.{ext}").read_text(encoding="utf-8"), encoding="utf-8")
            (dst_dir / f"solve.{ext}").unlink()
            seen.discard(f"solve.{ext}")
            seen.add(f"starter.{ext}")

    # drop files that no longer exist upstream (a re-run must not keep stale ones)
    for item in dst_dir.iterdir():
        if item.name in seen or item.name in SKIP_FILES:
            continue
        if item.is_file():
            item.unlink()
        elif item.name not in SKIP_DIRS:
            shutil.rmtree(item)
    return len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="old repository root")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--verify", type=int, default=0, help="verify N random migrated tasks")
    args = ap.parse_args()

    src_tasks = pathlib.Path(os.path.expanduser(args.src)) / "tasks"
    dst_tasks = HERE / "tasks"
    if not src_tasks.is_dir():
        sys.exit(f"no such directory: {src_tasks}")

    tasks, skipped = [], []
    for d in sorted(src_tasks.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if not meta.is_file():
            skipped.append((d.name, "no meta.json"))
            continue
        try:
            tid = json.loads(meta.read_text()).get("id") or d.name
        except Exception as e:
            skipped.append((d.name, f"bad meta.json: {e}"))
            continue
        if tid != d.name:
            # the directory must be named as the id, or the runner cannot find it
            skipped.append((d.name, f"id mismatch -> {tid}"))
            continue
        tasks.append(d)

    print(f"source     {src_tasks}")
    print(f"migratable {len(tasks)}   skipped {len(skipped)}")
    for name, why in skipped[:10]:
        print(f"  skip {name}: {why}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")

    # what would change
    touched = 0
    for d in tasks:
        for f in d.glob("*.py"):
            t = f.read_text(encoding="utf-8", errors="replace")
            if rewrite(t) != t:
                touched += 1
    print(f"python files needing an import rewrite: {touched}")

    if not args.apply:
        print("\ndry run — nothing written. re-run with --apply")
        return

    n = 0
    for d in tasks:
        copy_task(d, dst_tasks / d.name)
        n += 1
        if n % 250 == 0:
            print(f"  {n}/{len(tasks)}")
    print(f"migrated {n} tasks -> {dst_tasks}")

    if args.verify:
        sample = random.Random(0).sample(tasks, min(args.verify, len(tasks)))
        ok = fail = 0
        for d in sample:
            meta = json.loads((dst_tasks / d.name / "meta.json").read_text())
            script = "verify_native.sh" if meta.get("native") else "verify_task.sh"
            r = subprocess.run(["bash", str(HERE / "tools" / script), d.name],
                               capture_output=True, text=True, cwd=HERE, timeout=180)
            last = (r.stdout.strip().splitlines() or [""])[-1]
            if last == "TASK_OK":
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {d.name}: {last[:110]}")
        print(f"\nverified sample: {ok} OK, {fail} FAIL (of {len(sample)})")


if __name__ == "__main__":
    main()
