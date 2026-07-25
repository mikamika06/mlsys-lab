"""`mlsys` command-line: list tasks and grade a solver against a task."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, jsonsafe, runner
from .task import find_task, list_tasks

# dependency-free ANSI (disabled when not a tty)
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


GREEN = "32"
RED = "31"
AMBER = "33"
DIM = "2"
BOLD = "1"
STEEL = "36"


def _fmt(v: float) -> str:
    av = abs(v)
    if av != 0 and (av < 1e-3 or av >= 1e5):
        return f"{v:.3e}"
    return f"{v:.4g}"


def cmd_list(args) -> int:
    tasks = list_tasks(args.tasks_root)
    if not tasks:
        print(_c(DIM, f"no tasks under ./{args.tasks_root}"))
        return 0
    for t in tasks:
        m = t.meta
        tag = f"d{m.get('difficulty','?')} · {m.get('genre','?')} · {m.get('platform','?')}"
        print(f"  {_c(AMBER, m.get('id', t.path.name)):<40} {_c(DIM, tag)}")
        print(f"      {m.get('title','')}")
    return 0


def cmd_grade(args) -> int:
    task = find_task(args.task, args.tasks_root)
    res = runner.grade(task, solver_file=args.file)

    if args.json:
        out = {
            "task": res.task_id,
            "title": task.meta.get("title", res.task_id),
            "file": res.solver_file,
            "verdict": res.verdict,
            "passed": res.passed,
            "error": res.error,
            "metrics": res.metrics,
            "gates": [
                {"metric": g.metric, "op": g.op, "threshold": g.threshold,
                 "value": g.value, "ok": g.ok}
                for g in res.gates
            ],
        }
        print(jsonsafe.dumps(out, indent=2))
        return 0 if res.passed else 1

    print()
    print(f"  {_c(BOLD, task.meta.get('title', task.id))}")
    print(f"  {_c(DIM, task.id + ' · ' + args.file)}")
    print()

    if res.error:
        print(_c(RED, "  ERROR — solver or check raised:"))
        for line in res.error.rstrip().splitlines():
            print("    " + _c(DIM, line))
        print()
        print("  " + _c(RED, "● FAIL") + _c(DIM, "  (did not run)"))
        return 1

    name_w = max((len(g.metric) for g in res.gates), default=8)
    for g in res.gates:
        mark = _c(GREEN, "✓") if g.ok else _c(RED, "✗")
        col = GREEN if g.ok else RED
        gate_txt = f"{g.op} {_fmt(g.threshold)}"
        val_txt = _c(col, f"{_fmt(g.value):>12}")
        print(f"  {mark} {g.metric:<{name_w}}  {val_txt}   {_c(DIM, 'gate ' + gate_txt)}")

    info = [k for k in res.metrics if k not in {g.metric for g in res.gates}]
    for k in info:
        print(f"  {_c(DIM, 'i')} {k:<{name_w}}  {_c(STEEL, f'{_fmt(res.metrics[k]):>12}')}   {_c(DIM, 'info')}")

    print()
    n_ok = sum(1 for g in res.gates if g.ok)
    if res.passed:
        print("  " + _c(GREEN, f"● PASS") + _c(DIM, f"  {n_ok}/{len(res.gates)} gates"))
    else:
        print("  " + _c(RED, f"● FAIL") + _c(DIM, f"  {n_ok}/{len(res.gates)} gates"))
    print()
    return 0 if res.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlsys", description="Local auto-graded trainer.")
    p.add_argument("--version", action="version", version=f"mlsys {__version__}")
    p.add_argument("--tasks-root", default="tasks", help="directory holding task folders")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available tasks").set_defaults(func=cmd_list)

    g = sub.add_parser("grade", help="grade a solver against a task")
    g.add_argument("task", help="task id, dir name, or path")
    g.add_argument("--file", default="solve.py", help="solver file in the task dir")
    g.add_argument("--json", action="store_true", help="machine-readable output (for the editor extension)")
    g.set_defaults(func=cmd_grade)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
