"""Grader for `pyt-circular-import-failure-predictor`.

REAL ORACLE: every fixture graph is compiled to actual `.py` files on
disk and imported by a real CPython subprocess (`sys.executable`).
Whether each import succeeds or raises `ImportError` is observed from
real execution, not simulated or hardcoded -- this recomputes the
ground truth fresh on every grading run. All 24 fixtures are executed
in ONE subprocess (each case gets uniquely-namespaced module files, so
there's no `sys.modules` cross-contamination between cases), keeping
grading fast.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import shutil


# (modules, entry) -- modules: {name: [ops...]}, ops as documented in task.md
CASES = [
    ({'p': [('from_import', 'q', 'q_before')], 'q': [('bind', 'q_before'), ('import_module', 'p')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('import_module', 'q'), ('bind', 'p_after')], 'q': [('import_module', 'p'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after')], 'q': [('bind', 'q_before'), ('import_module', 'p'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('from_import', 'q', 'q_before')], 'q': [('bind', 'q_before'), ('import_module', 'p'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after')], 'q': [('bind', 'q_before'), ('from_import', 'p', 'p_before'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after'), ('bind', 'p_after')], 'q': [('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('import_module', 'q')], 'q': [('import_module', 'p')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('from_import', 'q', 'q_after')], 'q': [('import_module', 'p'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after')], 'q': [('from_import', 'p', 'p_before'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('bind', 'p_before'), ('import_module', 'q'), ('bind', 'p_after')], 'q': [('import_module', 'p')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_before'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('from_import', 'p', 'p_before')]}, 'p'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after'), ('bind', 'p_after')], 'q': [('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('from_import', 'q', 'q_after'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('import_module', 'p'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('import_module', 'q'), ('bind', 'p_after')], 'q': [('import_module', 'p')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('import_module', 'q'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_before'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('from_import', 'q', 'q_after')], 'q': [('bind', 'q_before'), ('import_module', 'p'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('bind', 'p_before'), ('import_module', 'q'), ('bind', 'p_after')], 'q': [('import_module', 'p')]}, 'p'),
    ({'p': [('import_module', 'q'), ('bind', 'p_after')], 'q': [('from_import', 'p', 'p_after')]}, 'p'),
    ({'p': [('bind', 'p_before'), ('from_import', 'q', 'q_after')], 'q': [('bind', 'q_before'), ('import_module', 'p'), ('bind', 'q_after')]}, 'q'),
    ({'p': [('import_module', 'q'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'p'),
    ({'p': [('from_import', 'q', 'q_before'), ('bind', 'p_after')], 'q': [('bind', 'q_before'), ('import_module', 'p')]}, 'p'),
    ({'p': [('import_module', 'q'), ('bind', 'p_after')], 'q': [('from_import', 'p', 'p_after'), ('bind', 'q_after')]}, 'p'),
]


def _ops_to_src(ops, name_map) -> str:
    lines = []
    for op in ops:
        if op[0] == "bind":
            lines.append(f"{op[1]} = 1")
        elif op[0] == "import_module":
            lines.append(f"import {name_map[op[1]]}")
        elif op[0] == "from_import":
            lines.append(f"from {name_map[op[1]]} import {op[2]}")
        else:
            raise ValueError(f"bad op {op!r}")
    return "\n".join(lines) + "\n"


def _real_oracle(cases) -> list:
    """Actually execute every case's import graph in one real CPython
    subprocess and return the list of True (succeeded) / False
    (ImportError) outcomes, in order."""
    d = tempfile.mkdtemp(prefix="circimp_")
    try:
        entry_names = []
        for i, (modules, entry) in enumerate(cases):
            name_map = {m: f"{m}{i}" for m in modules}
            for m, ops in modules.items():
                with open(os.path.join(d, f"{name_map[m]}.py"), "w") as f:
                    f.write(_ops_to_src(ops, name_map))
            entry_names.append(name_map[entry])

        driver = os.path.join(d, "_driver.py")
        with open(driver, "w") as f:
            f.write("import json\n")
            f.write(f"names = {entry_names!r}\n")
            f.write("out = []\n")
            f.write("for n in names:\n")
            f.write("    try:\n")
            f.write("        __import__(n)\n")
            f.write("        out.append(True)\n")
            f.write("    except ImportError:\n")
            f.write("        out.append(False)\n")
            f.write("print(json.dumps(out))\n")

        r = subprocess.run(
            [sys.executable, "_driver.py"], cwd=d,
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise RuntimeError(f"oracle driver failed: {r.stderr}")
        return json.loads(r.stdout.strip())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def grade(sol, fx) -> dict:
    expected = _real_oracle(CASES)   # fresh, real ground truth every run

    hits = 0
    for (modules, entry), exp in zip(CASES, expected):
        try:
            got = bool(sol.predict_import_result(copy.deepcopy(modules), entry))
        except Exception:
            got = None
        if got == exp:
            hits += 1

    return {"exact_match": hits / len(CASES)}
