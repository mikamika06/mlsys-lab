#!/usr/bin/env python3
"""A project is verified when the reference clears every milestone and the shipped
skeleton clears none of them. Same contract as tools/verify_task.sh, one level up:
without the second half a project can look green while grading nothing at all.

    python3 tools/verify_project.py [project-id ...]
"""
import concurrent.futures as cf
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mlsys.runners import project as pr  # noqa: E402

GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def grade_copy(pdir: str, src: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "w")
        shutil.copytree(src, work)
        return pr.grade(pdir, work)


def verify(pdir: str) -> tuple[bool, str]:
    spec = pr.load_spec(pdir)
    n = len(spec["milestones"])
    ref_dir = os.path.join(pdir, "reference")
    if not os.path.isdir(ref_dir):
        return False, "no reference/"

    ref = grade_copy(pdir, ref_dir)
    if ref["milestones_passed"] != n:
        bad = [f"{r['n']}" for r in ref["per_milestone"] if not r["passed"]]
        err = ref.get("error") or ""
        return False, f"reference clears {ref['milestones_passed']}/{n} (fails {','.join(bad)}) {err[:80]}"

    with tempfile.TemporaryDirectory() as tmp:
        work = pr.start(pdir, tmp)
        sk = pr.grade(pdir, work)
    if sk["milestones_passed"] != 0:
        return False, f"the skeleton already clears {sk['milestones_passed']}/{n} milestones"

    return True, f"reference {n}/{n}, skeleton 0/{n}"


def _one(pid):
    """A project is verified in its own process: a checker imports the learner's
    modules, and 1100 of those in one interpreter would collide on module names and
    leak state between units."""
    pdir = os.path.join(ROOT, "projects", pid)
    try:
        ok, msg = verify(pdir)
    except Exception as e:  # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    return pid, ok, msg


def main(argv):
    jobs = 0
    args = []
    for a in argv:
        if a.startswith("-j"):
            jobs = int(a[2:] or 0)
        else:
            args.append(a)
    root = os.path.join(ROOT, "projects")
    ids = args or sorted(d for d in os.listdir(root)
                         if os.path.isfile(os.path.join(root, d, "project.json")))
    if not jobs:
        jobs = min(8, max(1, (os.cpu_count() or 2) - 2)) if len(ids) > 2 else 1

    bad = 0
    if jobs == 1:
        results = [_one(p) for p in ids]
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_one, ids))
    for pid, ok, msg in results:
        mark = f"{GREEN}ok  {OFF}" if ok else f"{RED}FAIL{OFF}"
        print(f"  {mark} {pid:<44} {DIM}{msg}{OFF}")
        bad += 0 if ok else 1
    print(f"\n{len(ids) - bad}/{len(ids)} projects verified")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
