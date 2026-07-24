#!/usr/bin/env python3
"""Grade a REAL CUDA-C solve.cu.

The task's own check.py owns the fixture setup and the numpy oracle; it exposes
`grade(srcfile) -> dict` and, inside, parses the .cu with `mlsys.sim.cuda_c.CudaProgram`
and runs it thread-by-thread on the software GPU (`mlsys.sim.GPU`). This
runner is the thin shell around that: load check.py, call it, apply the gates from
meta.json, and print the same JSON shape `mlsys.runners.cpp` prints, so the
extension can treat both native tracks identically.
"""
import importlib.util
import json
import os
import sys


def _load_check(taskdir):
    spec = importlib.util.spec_from_file_location("mlsys_task_check",
                                                  os.path.join(taskdir, "check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(taskdir, srcfile="solve.cu"):
    meta = json.load(open(os.path.join(taskdir, "meta.json")))
    gates = meta.get("gates", [])
    try:
        check = _load_check(taskdir)
    except Exception as e:  # noqa: BLE001 — a broken task must not look like a wrong answer
        return {"passed": False, "error": "check.py failed to load: %s" % e, "metrics": {}}
    try:
        m = check.grade(srcfile)
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "error": "%s: %s" % (type(e).__name__, e), "metrics": {}}

    reported = []
    passed = True
    for g in gates:
        v = m.get(g["metric"])
        op, thr = g["op"], g["threshold"]
        ok = v is not None and (
            (v <= thr) if op == "<=" else (v >= thr) if op == ">=" else (v == thr))
        passed = passed and ok
        reported.append({"metric": g["metric"], "op": op, "threshold": thr, "value": v, "ok": ok})

    metrics = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    out = {"passed": passed, "metrics": metrics, "gates": reported}
    if m.get("error"):
        out["error"] = str(m["error"])        # e.g. a CUDA-C parse error, shown verbatim
    return out


if __name__ == "__main__":
    # `python -m mlsys.runners.cuda` already has the package importable; a task's
    # check.py imports `mlsys.sim` by name, so no path surgery is needed here.
    td = sys.argv[1]
    sf = sys.argv[2] if len(sys.argv) > 2 else "solve.cu"
    print(json.dumps(grade(td, sf), indent=2, default=str))
