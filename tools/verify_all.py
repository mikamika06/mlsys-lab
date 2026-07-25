#!/usr/bin/env python3
"""Verify the whole bank in parallel and write a report.

`TASK_OK` means the reference passes every gate AND the shipped starter fails at
least one. A task where the starter also passes teaches nothing and is treated as
broken, which is the failure mode this sweep exists to find.

    python3 tools/verify_all.py                 # everything, one worker per core
    python3 tools/verify_all.py --only cpp      # one track
    python3 tools/verify_all.py --jobs 4
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / ".internal" / "verify_report.json"


def verify(tid):
    meta = json.loads((ROOT / "tasks" / tid / "meta.json").read_text())
    script = "verify_native.sh" if meta.get("native") else "verify_task.sh"
    t0 = time.time()
    try:
        r = subprocess.run(["bash", str(ROOT / "tools" / script), tid],
                           capture_output=True, text=True, cwd=ROOT, timeout=240)
        out = (r.stdout.strip().splitlines() or [""])[-1]
    except subprocess.TimeoutExpired:
        out = "TASK_FAIL: timeout after 240s"
    return {"id": tid, "native": meta.get("native") or "python",
            "genre": meta.get("genre"), "ok": out == "TASK_OK",
            "msg": "" if out == "TASK_OK" else out[:160],
            "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["python", "cpp", "cuda"])
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    ids = []
    for d in sorted((ROOT / "tasks").iterdir()):
        mf = d / "meta.json"
        if not mf.is_file():
            continue
        try:
            nat = json.loads(mf.read_text()).get("native") or "python"
        except Exception:
            continue
        if a.only and nat != a.only:
            continue
        ids.append(d.name)

    print(f"verifying {len(ids)} tasks with {a.jobs} workers", flush=True)
    t0, done, res = time.time(), 0, []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(verify, ids):
            res.append(r)
            done += 1
            if done % 100 == 0:
                bad = sum(1 for x in res if not x["ok"])
                rate = done / max(1e-9, time.time() - t0)
                print(f"  {done}/{len(ids)}  failing {bad}  "
                      f"eta {int((len(ids)-done)/max(rate,1e-9)/60)}m", flush=True)

    bad = [r for r in res if not r["ok"]]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(res, indent=1))

    print(f"\n{len(res)-len(bad)}/{len(res)} OK   ({len(bad)} failing)   "
          f"{int(time.time()-t0)}s")
    by_track, by_reason = {}, {}
    for r in bad:
        by_track[r["native"]] = by_track.get(r["native"], 0) + 1
        key = r["msg"].split(":")[1].strip()[:52] if ":" in r["msg"] else r["msg"][:52]
        by_reason[key] = by_reason.get(key, 0) + 1
    if bad:
        print("\nby track: ", by_track)
        print("by reason:")
        for k, v in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"  {v:>4}  {k}")
        print("\nfirst 25:")
        for r in bad[:25]:
            print(f"  {r['id'][:56]:58} {r['msg'][:70]}")
    print(f"\nfull report: {REPORT}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
