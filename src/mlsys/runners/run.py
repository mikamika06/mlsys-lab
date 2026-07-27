#!/usr/bin/env python3
"""Execute a candidate file and show what it does — not what it scores.

`mlsys grade` answers exactly one question: do the gates pass. It says nothing
about a print, a shape, a compiler diagnostic or a traceback, which is most of
what the working half of an exercise actually needs. This runs the same file the
grader would read and shows whatever comes out.

    python -m mlsys.runners.run <taskdir> <candidate>

Output is never captured: the child writes straight through to this process's
stdout and stderr, so a caller reading a pipe (the editor does) sees a slow loop
as it happens rather than after it ends. The exit status is the program's own.

Each track runs the way that track is actually run:

    python   the file, as a script
    cpp      clang++ against the task's own driver, then the binary
    cuda     parsed and executed on the software GPU, with the task's fixtures

Nothing here applies a gate or reports a verdict. That is grading, and grading
is the other command.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

CXX = os.environ.get("MLSYS_CXX", "clang++")
FLAGS = ["-O2", "-std=c++20"]

# A candidate that defines its own main() is a program and runs alone; one that
# does not is an implementation of the task's contract and needs the task's
# driver to call it. Both are legitimate things to press Run on.
HAS_MAIN = re.compile(r"^\s*(?:int|auto)\s+main\s*\(", re.M)


def _meta(taskdir: str) -> dict:
    try:
        with open(os.path.join(taskdir, "meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — a missing meta.json is a python task
        return {}


def _echo(*parts: str) -> None:
    print("$ " + " ".join(parts), flush=True)


def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def run_python(taskdir: str, cand: str) -> int:
    _echo("python", os.path.basename(cand))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (taskdir, env.get("PYTHONPATH")) if p)
    return subprocess.call([sys.executable, cand], cwd=os.path.dirname(cand) or ".", env=env)


def run_cpp(taskdir: str, cand: str, meta: dict) -> int:
    own_main = bool(HAS_MAIN.search(_read(cand)))
    driver = os.path.join(taskdir, "main.cpp")
    if not own_main and not os.path.isfile(driver):
        print(f"{os.path.basename(cand)} defines no main() and this task ships no driver "
              f"to link it with.", file=sys.stderr)
        return 2
    sources = [cand] if own_main else [driver, cand]
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "a.out")
        cmd = [CXX, *FLAGS, *(meta.get("cxx_flags") or []), "-I", taskdir, "-o", exe, *sources]
        _echo(CXX, *FLAGS, *(meta.get("cxx_flags") or []),
              "-I", os.path.basename(taskdir),
              *(os.path.basename(s) for s in sources))
        if not own_main:
            print(f"  no main() in {os.path.basename(cand)}, so it is linked against the "
                  f"task's own driver", flush=True)
        cc = subprocess.run(cmd, capture_output=True, text=True)
        # Warnings are worth reading even when the compile succeeded — this is the
        # one place a learner sees them at all.
        if cc.stderr.strip():
            print(cc.stderr.rstrip(), file=sys.stderr, flush=True)
        if cc.returncode != 0:
            return cc.returncode
        _echo("./a.out")
        return subprocess.call([exe], cwd=os.path.dirname(cand) or ".")


def _load_check(taskdir: str):
    spec = importlib.util.spec_from_file_location("mlsys_task_check",
                                                  os.path.join(taskdir, "check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cuda(taskdir: str, cand: str) -> int:
    """There is no hardware and no host program: a .cu is parsed, then executed
    thread by thread on the software GPU. The fixtures and the launch shape live
    in the task's own check.py, which is the only thing that knows them, so the
    run is that same execution with the gates left off."""
    from ..sim import CudaProgram

    _echo("software GPU", os.path.basename(cand))
    try:
        prog = CudaProgram(_read(cand))
    except ValueError as e:                      # includes CudaParseError
        print(str(e), file=sys.stderr, flush=True)
        return 1
    print("  kernels: " + ", ".join(sorted(prog.funcs)), flush=True)

    try:
        check = _load_check(taskdir)
    except Exception as e:  # noqa: BLE001
        print(f"the task's check.py failed to load: {e}", file=sys.stderr)
        return 2
    try:
        m = check.grade(os.path.abspath(cand))
    except Exception as e:  # noqa: BLE001 — a fault in the kernel is the point of running
        print(f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1

    err = m.pop("error", None)
    width = max((len(k) for k in m), default=0)
    for k in sorted(m):
        v = m[k]
        print(f"  {k.ljust(width)} = {v}", flush=True)
    if err:
        print(str(err), file=sys.stderr, flush=True)
        return 1
    print("  no gates applied — `mlsys grade` is the verdict", flush=True)
    return 0


def run(taskdir: str, cand: str) -> int:
    taskdir = os.path.abspath(taskdir)
    cand = os.path.abspath(cand)
    if not os.path.isfile(cand):
        print(f"no such file: {cand}", file=sys.stderr)
        return 2
    meta = _meta(taskdir)
    nat = meta.get("native")
    if nat == "cpp":
        return run_cpp(taskdir, cand, meta)
    if nat == "cuda":
        return run_cuda(taskdir, cand)
    return run_python(taskdir, cand)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(sys.argv[1], sys.argv[2]))
