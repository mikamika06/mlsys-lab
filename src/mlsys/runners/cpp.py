#!/usr/bin/env python3
"""Grade a REAL C++ task: compile the learner's solve.cpp with the local clang++,
run it against a fixed driver, and compare the output to a real C++ reference
solution (oracle-first). No Python simulation — actual compilation & execution.

Task dir: main.cpp (driver), sol.hpp (contract), ref.cpp (reference),
solve.cpp (candidate), meta.json (gates).
"""
import subprocess, os, sys, json, re, tempfile

CXX = os.environ.get("ARENA_CXX", "clang++")
FLAGS = ["-O2", "-std=c++20"]


def _compile_run(taskdir, srcfile, timeout=15):
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "a.out")
        cc = subprocess.run([CXX, *FLAGS, "-I", taskdir, "-o", exe,
                             os.path.join(taskdir, "main.cpp"), os.path.join(taskdir, srcfile)],
                            capture_output=True, text=True)
        if cc.returncode != 0:
            return None, "COMPILE ERROR:\n" + cc.stderr[:800]
        try:
            run = subprocess.run([exe], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT (infinite loop?)"
        if run.returncode != 0:
            return None, "RUNTIME ERROR (exit %d):\n%s" % (run.returncode, run.stderr[:400])
        return run.stdout.strip(), None


def _nums(s):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)]


def grade(taskdir, srcfile="solve.cpp"):
    meta = json.load(open(os.path.join(taskdir, "meta.json")))
    gates = meta.get("gates", [])
    ref, e = _compile_run(taskdir, "ref.cpp")
    if ref is None:
        return {"passed": False, "error": "reference " + e, "metrics": {}}
    cand, e = _compile_run(taskdir, srcfile)
    if cand is None:
        return {"passed": False, "error": e, "metrics": {}}
    m = {}
    rn, cn = _nums(ref), _nums(cand)
    if rn and len(rn) == len(cn):
        m["max_abs_err"] = max((abs(a - b) for a, b in zip(rn, cn)), default=0.0)
    m["exact_match"] = 1.0 if cand == ref else 0.0
    passed = all(m.get(g["metric"]) is not None and (
        (m[g["metric"]] <= g["threshold"]) if g["op"] == "<=" else
        (m[g["metric"]] >= g["threshold"]) if g["op"] == ">=" else
        (m[g["metric"]] == g["threshold"])) for g in gates)
    return {"passed": passed, "metrics": m, "gates": gates}


if __name__ == "__main__":
    print(json.dumps(grade(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "solve.cpp"), indent=2))
