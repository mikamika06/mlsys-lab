"""`mlsys` command-line: list tasks and grade a solver against a task."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, jsonsafe, runner
from .bank import bank_root
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
    try:
        root = bank_root(args.tasks_root)
    except FileNotFoundError as e:
        print(_c(RED, str(e)))
        return 2
    tasks = list_tasks(root)
    if not tasks:
        print(_c(DIM, f"no tasks under {root}"))
        return 0
    for t in tasks:
        m = t.meta
        tag = f"d{m.get('difficulty','?')} · {m.get('genre','?')} · {m.get('platform','?')}"
        print(f"  {_c(AMBER, m.get('id', t.path.name)):<40} {_c(DIM, tag)}")
        print(f"      {m.get('title','')}")
    return 0


# The learner's file. `starter.py`/`starter.cpp`/`starter.cu` is what ships; the
# candidate is always their own copy, and it never goes inside the task directory
# because an installed bank lives in site-packages.
SRCFILE = {None: "solve.py", "cpp": "solve.cpp", "cuda": "solve.cu"}
STARTER = {None: "starter.py", "cpp": "starter.cpp", "cuda": "starter.cu"}


def _native(task):
    return task.meta.get("native") or None


def _find_candidate(task, given):
    """Where the learner's attempt is. `given` wins; otherwise look in the obvious
    places, nearest first, so `mlsys grade <id>` works after `mlsys start <id>`."""
    want = SRCFILE[_native(task)]
    # Always resolved: the runners join the candidate onto the task directory, and
    # pathlib only discards the left operand for an ABSOLUTE right one. A relative
    # path handed back from here became <task-dir>/<relative-path> and vanished.
    if given:
        p = Path(given).expanduser()
        # a bare filename may name a file in cwd or in the task dir (a checkout)
        for cand in (p, Path.cwd() / p, task.path / p):
            if cand.is_file():
                return cand.resolve()
        return None
    for cand in (Path.cwd() / task.id / want, Path.cwd() / want, task.path / want):
        if cand.is_file():
            return cand.resolve()
    return None


def cmd_start(args) -> int:
    """Copy a task's starter into a working directory so there is something to edit."""
    try:
        task = find_task(args.task, args.tasks_root)
    except FileNotFoundError as e:
        print(_c(RED, str(e)))
        return 2
    nat = _native(task)
    src = task.path / STARTER[nat]
    if not src.is_file():
        print(_c(RED, f"{task.id} ships no {STARTER[nat]}"))
        return 2
    dest_dir = Path(args.dir).expanduser() / task.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / SRCFILE[nat]
    if dest.exists() and not args.force:
        print(_c(AMBER, f"{dest} exists (use --force to overwrite)"))
    else:
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  {_c(BOLD, task.meta.get('title', task.id))}")
        print(f"  {_c(DIM, 'wrote')} {dest}")
    stmt = task.path / "task.md"
    if stmt.is_file():
        print(f"  {_c(DIM, 'read')}  {stmt}")
    if nat == "cpp":
        print(f"  {_c(DIM, 'contract')} {task.path / 'sol.hpp'}")
    print()
    print(f"  {_c(STEEL, 'mlsys grade ' + task.id)}")
    return 0


def cmd_grade(args) -> int:
    try:
        task = find_task(args.task, args.tasks_root)
    except FileNotFoundError as e:
        print(_c(RED, str(e)))
        return 2
    cand = _find_candidate(task, args.file)
    if cand is None:
        want = SRCFILE[_native(task)]
        print(_c(RED, f"no {want} found for {task.id}."))
        print(_c(DIM, f"  it is a {_native(task) or 'python'} task, so your file must be {want}."))
        print()
        print(f"  {_c(STEEL, 'mlsys start ' + task.id)}   {_c(DIM, 'writes the starter here to edit')}")
        print(f"  {_c(DIM, 'or:')} mlsys grade {task.id} --file path/to/{want}")
        return 2

    nat = _native(task)
    if nat in ("cpp", "cuda"):
        # A native task is compiled or executed by its own runner; the python
        # check.py path cannot grade it, and used to fail with a confusing
        # "solve.py not found" for all 460 of them.
        from importlib import import_module
        raw = import_module(f".runners.{nat}", __package__).grade(str(task.path), str(cand))
        res = runner.GradeResult(task_id=task.id, solver_file=str(cand),
                                 metrics=raw.get("metrics") or {},
                                 passed=bool(raw.get("passed")),
                                 error=raw.get("error"))
        for g in task.gates:
            v = res.metrics.get(g["metric"])
            ok = v is not None and (v <= g["threshold"] if g["op"] == "<=" else
                                    v >= g["threshold"] if g["op"] == ">=" else
                                    v == g["threshold"])
            res.gates.append(runner.GateResult(g["metric"], g["op"], g["threshold"], v, bool(ok)))
    else:
        res = runner.grade(task, solver_file=str(cand))

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
    print(f"  {_c(DIM, task.id + ' · ' + str(cand))}")
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
    p.add_argument("--tasks-root", default=None,
                   help="directory holding task folders (default: ./tasks in a checkout, "
                        "else the bank installed with the package; see $MLSYS_TASKS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available tasks").set_defaults(func=cmd_list)

    st = sub.add_parser("start", help="copy a task's starter into a working directory")
    st.add_argument("task", help="task id, dir name, or path")
    st.add_argument("--dir", default=".", help="where to create <task-id>/ (default: here)")
    st.add_argument("--force", action="store_true", help="overwrite an existing attempt")
    st.set_defaults(func=cmd_start)

    g = sub.add_parser("grade", help="grade a solver against a task")
    g.add_argument("task", help="task id, dir name, or path")
    g.add_argument("--file", default=None,
                   help="your solution file (default: ./<task-id>/solve.* or ./solve.*)")
    g.add_argument("--json", action="store_true", help="machine-readable output (for the editor extension)")
    g.set_defaults(func=cmd_grade)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
