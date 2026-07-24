#!/usr/bin/env python3
"""Keep the bank clean while workers run: remove tasks/<id>/ folders that FAIL
verify (broken builds the old worker left behind). Verifies before removing;
skips anything currently claimed/in-flight. Safe to run alongside the workers.
"""
import glob, os, shutil, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def sweep():
    failed = {os.path.basename(p)[:-5] for p in glob.glob("tools/specs_failed/*.spec")}
    claimed = {os.path.basename(p)[:-5] for p in glob.glob("tools/specs_claimed/*.spec")}
    bank = {os.path.basename(os.path.dirname(p)) for p in glob.glob("tasks/*/meta.json")}
    removed = 0
    for tid in (bank & failed) - claimed:
        r = subprocess.run(["bash", "tools/verify_task.sh", tid], capture_output=True, text=True).stdout
        if not ("TASK_OK" in r and "TASK_FAIL" not in r):
            shutil.rmtree(os.path.join("tasks", tid), ignore_errors=True)
            removed += 1
    return removed


while True:
    n = sweep()
    if n:
        print(f"{time.strftime('%H:%M:%S')} janitor removed {n} broken tasks", flush=True)
    time.sleep(90)
