"""Grade a solver against a task's gates.

check.py in each task defines `grade(solver_module, fixtures) -> dict[str, float]`.
The runner compares those metric values to the declarative gates in meta.json and
AND-aggregates them into a verdict.
"""
from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field

from .task import Task

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


@dataclass
class GateResult:
    metric: str
    op: str
    threshold: float
    value: float
    ok: bool


@dataclass
class GradeResult:
    task_id: str
    solver_file: str
    metrics: dict = field(default_factory=dict)
    gates: list = field(default_factory=list)
    passed: bool = False
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        return "PASS" if self.passed else "FAIL"


def grade(task: Task, solver_file: str = "solve.py") -> GradeResult:
    res = GradeResult(task_id=task.id, solver_file=solver_file)
    try:
        fixtures = task.load_fixtures()
        solver = task.load_module(solver_file)
        check = task.load_module("check.py")
        metrics = check.grade(solver, fixtures)
        res.metrics = {k: float(v) for k, v in metrics.items()}
    except Exception:  # noqa: BLE001 — any solver/check failure is a red verdict
        res.error = traceback.format_exc(limit=6)
        return res

    all_ok = True
    for g in task.gates:
        val = res.metrics.get(g["metric"], math.nan)
        op = g["op"]
        thr = float(g["threshold"])
        ok = (not math.isnan(val)) and _OPS[op](val, thr)
        all_ok = all_ok and ok
        res.gates.append(GateResult(g["metric"], op, thr, val, ok))
    res.passed = all_ok and len(task.gates) > 0
    return res
