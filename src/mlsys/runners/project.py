from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys

from .. import jsonsafe


def load_spec(projdir: str) -> dict:
    with open(os.path.join(projdir, "project.json"), encoding="utf-8") as f:
        return json.load(f)


def _load_check(path: str):
    spec = importlib.util.spec_from_file_location("mlsys_project_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def start(projdir: str, dest_dir: str, force: bool = False) -> str:
    spec = load_spec(projdir)
    dest = os.path.join(os.path.abspath(dest_dir), spec["id"])
    src = os.path.join(projdir, "skeleton")
    if os.path.isdir(dest) and not force:
        return dest
    os.makedirs(dest, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = os.path.join(dest, rel) if rel != "." else dest
        os.makedirs(target, exist_ok=True)
        for fn in files:
            if fn.endswith(".pyc"):
                continue
            dst = os.path.join(target, fn)
            if force or not os.path.exists(dst):
                shutil.copy2(os.path.join(root, fn), dst)
    brief = os.path.join(projdir, "brief.md")
    if os.path.isfile(brief):
        shutil.copy2(brief, os.path.join(dest, "brief.md"))
    return dest


def _gate_ok(value, op: str, thr) -> bool:
    if value is None:
        return False
    try:
        if op == "<=":
            return value <= thr
        if op == ">=":
            return value >= thr
        if op == "<":
            return value < thr
        if op == ">":
            return value > thr
        return value == thr
    except TypeError:
        return False


def grade_milestone(projdir: str, workdir: str, m: dict) -> dict:
    check_path = os.path.join(projdir, m["check"])
    if not os.path.isfile(check_path):
        return {"passed": False, "error": f"missing checker {m['check']}", "metrics": {}, "gates": []}
    harness_dir = os.path.dirname(os.path.abspath(check_path))
    sys.path.insert(0, harness_dir)
    sys.path.insert(0, os.path.abspath(workdir))
    try:
        mod = _load_check(check_path)
        metrics = mod.check(os.path.abspath(workdir))
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "error": f"{type(e).__name__}: {e}", "metrics": {}, "gates": []}
    finally:
        for p in (os.path.abspath(workdir), harness_dir):
            if p in sys.path:
                sys.path.remove(p)
        for name in [n for n, mo in sys.modules.items()
                     if getattr(mo, "__file__", None)
                     and os.path.abspath(workdir) in str(getattr(mo, "__file__", ""))]:
            del sys.modules[name]

    note = metrics.pop("_note", None)
    gates = []
    passed = True
    for g in m.get("gates", []):
        v = metrics.get(g["metric"])
        ok = _gate_ok(v, g["op"], g["threshold"])
        passed = passed and ok
        gates.append({"metric": g["metric"], "op": g["op"], "threshold": g["threshold"],
                      "value": v, "ok": bool(ok)})
    out = {"passed": passed, "metrics": {k: v for k, v in metrics.items()
                                         if isinstance(v, (int, float, bool))},
           "gates": gates, "milestone": m["n"], "title": m["title"]}
    if note:
        out["note"] = str(note)
    return out


def grade(projdir: str, workdir: str, milestone: int | None = None) -> dict:
    spec = load_spec(projdir)
    ms = spec["milestones"]
    if milestone is not None:
        sel = [m for m in ms if m["n"] == milestone]
        if not sel:
            return {"passed": False, "error": f"no milestone {milestone}", "metrics": {}, "gates": []}
        r = grade_milestone(projdir, workdir, sel[0])
        r["project"] = spec["id"]
        return r

    results = []
    for m in ms:
        r = grade_milestone(projdir, workdir, m)
        results.append(r)
        if not r["passed"]:
            break
    done = sum(1 for r in results if r["passed"])
    last = results[-1] if results else {"metrics": {}, "gates": []}
    return {
        "project": spec["id"],
        "passed": done == len(ms),
        "milestones_passed": done,
        "milestones_total": len(ms),
        "metrics": dict(last.get("metrics", {}), milestones_passed=done),
        "gates": last.get("gates", []),
        "error": last.get("error"),
        "milestone": last.get("milestone"),
        "title": last.get("title"),
        "note": last.get("note"),
        "per_milestone": [{"n": r.get("milestone"), "title": r.get("title"),
                           "passed": r["passed"], "error": r.get("error")} for r in results],
    }


if __name__ == "__main__":
    pd, wd = sys.argv[1], sys.argv[2]
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(jsonsafe.dumps(grade(pd, wd, ms), indent=2))
