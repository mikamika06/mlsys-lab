"""Where the task bank lives.

The bank is 2052 directories under `tasks/`. In a git checkout that is `./tasks`
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


def curriculum_file() -> Path | None:
    """The area/sub-track grouping for the bank, if it shipped or is in the checkout.

    Editors group 2052 tasks by it. Missing is survivable — the id prefix recovers
    the 14 areas — so this returns None rather than raising.
    """
    p = Path(__file__).resolve().parent / "task_list2.json"
    return p if p.is_file() else None


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
