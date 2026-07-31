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

from mlsys import bank  # noqa: E402
from mlsys.runners import project as pr  # noqa: E402

GREEN, RED, AMBER, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def grade_copy(pdir: str, src: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "w")
        shutil.copytree(src, work)
        return pr.grade(pdir, work)


class Skipped(Exception):
    """A T1 unit needs a real library. Absent, its reference cannot clear its own
    milestones — which is a fact about the machine, not about the unit. Reporting
    that as a failure trains everyone to ignore a red check."""


def verify(pdir: str) -> tuple[bool, str]:
    spec = pr.load_spec(pdir)
    n = len(spec["milestones"])

    lack = bank.missing_pkgs(spec)
    if lack:
        raise Skipped(", ".join(lack))

    # A unit without a ticket is not a unit. The panel shows nothing, the CLI shows
    # nothing, and the learner is handed a skeleton with no statement of the problem
    # — which the milestone check alone happily called green.
    brief = os.path.join(pdir, "brief.md")
    if not os.path.isfile(brief):
        return False, "no brief.md"
    with open(brief, encoding="utf-8") as f:
        words = len(f.read().split())
    if words < 80:
        return False, f"brief.md is {words} words, too thin to be a ticket"
    for key in ("id", "title", "area", "tier"):
        if not spec.get(key):
            return False, f"project.json has no {key}"

    ref_dir = os.path.join(pdir, "reference")
    if not os.path.isdir(ref_dir):
        return False, "no reference/"
    if not any(f.endswith(".py") for _, _, fs in os.walk(ref_dir) for f in fs):
        return False, "reference/ is empty"

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
        return pid, "ok" if ok else "fail", msg
    except Skipped as e:
        return pid, "skip", f"needs {e}"
    except Exception as e:  # noqa: BLE001
        return pid, "fail", f"{type(e).__name__}: {e}"


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

    if jobs == 1:
        results = [_one(p) for p in ids]
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_one, ids))
    bad = skipped = 0
    for pid, state, msg in results:
        mark = {"ok": f"{GREEN}ok  {OFF}", "skip": f"{AMBER}skip{OFF}"}.get(state, f"{RED}FAIL{OFF}")
        print(f"  {mark} {pid:<44} {DIM}{msg}{OFF}")
        bad += state == "fail"
        skipped += state == "skip"
    done = len(ids) - bad - skipped
    tail = f", {skipped} skipped for a missing package" if skipped else ""
    print(f"\n{done}/{len(ids) - skipped} projects verified{tail}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
