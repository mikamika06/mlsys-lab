"""Where the task bank lives.

The bank is a directory of task directories under `tasks/`. In a git checkout that is `./tasks`
relative to the repo root; in a `pip install mlsys-lab` it is shipped inside the
package as `mlsys/tasks` (hatchling force-includes it — see pyproject.toml). Code
must not assume either one, so everything asks here.

Resolution order, first hit wins:

  1. `$MLSYS_TASKS`             an explicit override, for a second bank or a fork
  2. `./tasks` upward from cwd  a checkout, so a developer's edits are what runs
  3. `mlsys/tasks`             the copy installed with the package

A checkout beats the installed copy on purpose: someone working in the repo who
also has the package installed means to grade the tasks they are editing.
"""
from __future__ import annotations

import os
from pathlib import Path

_MARKER = "meta.json"


def _looks_like_bank(p: Path) -> bool:
    """A directory holding at least one task, not merely a directory named tasks."""
    if not p.is_dir():
        return False
    return any(child.joinpath(_MARKER).is_file() for child in p.iterdir() if child.is_dir())


def bundled_root() -> Path:
    """The bank shipped inside the installed package (may not exist in a checkout)."""
    return Path(__file__).resolve().parent / "tasks"


# A handful of tasks need a package this project does not depend on: 11 use scipy in
# their oracle, 3 use ml_dtypes, one mpmath, one torch. Depending on all of those would
# mean every learner installing torch to get the bank, so they are declared per task in
# meta.json as `requires_pkgs` and installable as extras. Import names, not pip names —
# `ml_dtypes` imports that way and installs as `ml-dtypes`.
PIP_NAME = {"ml_dtypes": "ml-dtypes"}


def missing_pkgs(meta: dict) -> list[str]:
    """Which of a task's declared requirements cannot be imported here.

    Returned as pip names, ready to paste after `pip install`. Checked with
    importlib.util.find_spec so nothing is actually imported and no import side
    effect lands in the grading process.
    """
    import importlib.util

    out = []
    for mod in meta.get("requires_pkgs", []):
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(PIP_NAME.get(mod, mod))
    return out


def curriculum_file() -> Path | None:
    """The area/sub-track grouping for the bank, if it shipped or is in the checkout.

    Editors group the bank by it. Missing is survivable — the id prefix recovers
    the 14 areas — so this returns None rather than raising.
    """
    p = Path(__file__).resolve().parent / "task_list2.json"
    return p if p.is_file() else None


def projects_root(explicit: str | os.PathLike | None = None) -> Path | None:
    """Where the Part-2 projects live, or None. Same resolution order as the bank:
    an explicit path, then $MLSYS_PROJECTS, then ./projects upward from cwd, then
    what shipped inside the package."""
    def ok(p: Path) -> bool:
        return p.is_dir() and any(c.joinpath("project.json").is_file() for c in p.iterdir()
                                  if c.is_dir())

    if explicit is not None:
        p = Path(explicit).expanduser()
        return p.resolve() if ok(p) else None
    env = os.environ.get("MLSYS_PROJECTS")
    if env:
        p = Path(env).expanduser()
        return p.resolve() if ok(p) else None
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        cand = d / "projects"
        if ok(cand):
            return cand
    b = Path(__file__).resolve().parent / "projects"
    return b if ok(b) else None


def find_project(name: str, explicit: str | os.PathLike | None = None) -> Path:
    p = Path(name).expanduser()
    if (p / "project.json").is_file():
        return p.resolve()
    root = projects_root(explicit)
    if root is not None and (root / name / "project.json").is_file():
        return (root / name).resolve()
    raise FileNotFoundError(f"no project named {name}")


def list_projects(explicit: str | os.PathLike | None = None) -> list[Path]:
    root = projects_root(explicit)
    if root is None:
        return []
    return sorted(c for c in root.iterdir() if (c / "project.json").is_file())


def bank_root(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the task bank. Raises FileNotFoundError with a fixable message."""
    if explicit is not None:
        p = Path(explicit).expanduser()
        if _looks_like_bank(p):
            return p.resolve()
        raise FileNotFoundError(f"no tasks under {p} (expected directories each holding {_MARKER})")

    env = os.environ.get("MLSYS_TASKS")
    if env:
        p = Path(env).expanduser()
        if _looks_like_bank(p):
            return p.resolve()
        raise FileNotFoundError(f"MLSYS_TASKS={env} does not hold a task bank")

    # a checkout: ./tasks, or the tasks/ of a repo we are somewhere inside
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        cand = d / "tasks"
        if _looks_like_bank(cand):
            return cand

    b = bundled_root()
    if _looks_like_bank(b):
        return b

    raise FileNotFoundError(
        "no task bank found. Either run inside a mlsys-lab checkout, or "
        "`pip install mlsys-lab` to get the bank, or point MLSYS_TASKS at one."
    )
