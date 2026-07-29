#!/usr/bin/env python3
"""Verify every complete project and group the failures by cause.

At one unit a failure is a message; at four hundred it has to be a histogram, or
the same defect gets fixed four hundred times by hand.

    python3 tools/triage_projects.py            # everything complete
    python3 tools/triage_projects.py --incomplete   # also list half-written units
"""
import argparse
import collections
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.join(ROOT, "projects")

CAUSE = [
    (re.compile(r"skeleton already clears"), "gate measures nothing (skeleton passes)"),
    (re.compile(r"reference clears \d+/"), "reference does not clear its own milestones"),
    (re.compile(r"missing checker"), "a milestone points at a checker that is not there"),
    (re.compile(r"no reference/"), "no reference at all"),
    (re.compile(r"ModuleNotFoundError|ImportError"), "import fails in the checker"),
    (re.compile(r"KeyError|AttributeError|TypeError"), "checker crashes on its own metrics"),
    (re.compile(r"needs .* pip install|_note.*needs"), "declared package is absent here"),
]


def complete(pid):
    d = os.path.join(PROJ, pid)
    return (os.path.isfile(os.path.join(d, "project.json"))
            and os.path.isdir(os.path.join(d, "harness"))
            and os.path.isdir(os.path.join(d, "reference"))
            and os.path.isdir(os.path.join(d, "skeleton")))


def classify(msg):
    for rx, name in CAUSE:
        if rx.search(msg):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incomplete", action="store_true")
    ap.add_argument("-j", type=int, default=0)
    a = ap.parse_args()

    if not os.path.isdir(PROJ):
        print("no projects/")
        return 0
    ids = sorted(d for d in os.listdir(PROJ) if os.path.isdir(os.path.join(PROJ, d)))
    ready = [i for i in ids if complete(i)]
    half = [i for i in ids if i not in ready]

    print(f"{len(ids)} directories · {len(ready)} complete · {len(half)} half-written")
    if a.incomplete and half:
        for i in half[:40]:
            have = [x for x in ("project.json", "harness", "reference", "skeleton")
                    if os.path.exists(os.path.join(PROJ, i, x))]
            print(f"    {i:<52} has {', '.join(have) or 'nothing'}")
    if not ready:
        return 0

    cmd = [sys.executable, os.path.join(ROOT, "tools", "verify_project.py")]
    if a.j:
        cmd.append(f"-j{a.j}")
    r = subprocess.run(cmd + ready, capture_output=True, text=True, cwd=ROOT)
    fails = collections.defaultdict(list)
    ok = 0
    for line in r.stdout.splitlines():
        clean = re.sub(r"\033\[[0-9;]*m", "", line)
        if clean.strip().startswith("ok "):
            ok += 1
        elif clean.strip().startswith("FAIL"):
            parts = clean.split(None, 2)
            pid = parts[1] if len(parts) > 1 else "?"
            msg = parts[2] if len(parts) > 2 else ""
            fails[classify(msg)].append((pid, msg.strip()))

    print(f"\nverified {ok}/{len(ready)}")
    if fails:
        print("\nfailures by cause:")
        for cause, rows in sorted(fails.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  {len(rows):4}  {cause}")
            for pid, msg in rows[:5]:
                print(f"        {pid:<50} {msg[:80]}")
            if len(rows) > 5:
                print(f"        … {len(rows) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
