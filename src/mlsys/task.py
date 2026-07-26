"""Task loading: meta.json, fixtures (.npy arrays), and solver/check modules."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from .bank import bank_root


class Task:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        meta_file = self.path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"no meta.json in {self.path}")
        self.meta = json.loads(meta_file.read_text())
        self.id = self.meta.get("id", self.path.name)

    # -- fixtures: hidden holdout stored as arrays, never models --
    def load_fixtures(self) -> dict[str, np.ndarray]:
        fx: dict[str, np.ndarray] = {}
        for f in self.meta.get("fixtures", []):
            arr = np.load(self.path / "fixtures" / f, allow_pickle=False)
            fx[Path(f).stem] = arr
        return fx

    # -- import a python file from the task dir as a fresh module --
    def load_module(self, filename: str):
        p = self.path / filename
        if not p.exists():
            raise FileNotFoundError(f"{filename} not found in task {self.id}")
        mod_name = f"mlsys_task_{self.id.replace('-', '_')}_{Path(filename).stem}"
        spec = importlib.util.spec_from_file_location(mod_name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    @property
    def gates(self) -> list[dict]:
        return self.meta.get("gates", [])


def find_task(name: str, tasks_root: str | Path | None = None) -> Task:
    """Resolve a task by path, by directory name under the bank, or by meta id."""
    p = Path(name)
    if (p / "meta.json").exists():
        return Task(p)
    root = bank_root(tasks_root)
    if (root / name / "meta.json").exists():
        return Task(root / name)
    for d in sorted(root.glob("*")):
        mf = d / "meta.json"
        if mf.exists():
            try:
                if json.loads(mf.read_text()).get("id") == name:
                    return Task(d)
            except json.JSONDecodeError:
                continue
    raise FileNotFoundError(f"task '{name}' not found (looked in {root})")


def list_tasks(tasks_root: str | Path | None = None) -> list[Task]:
    root = bank_root(tasks_root)
    out = []
    for d in sorted(root.glob("*")):
        if (d / "meta.json").exists():
            out.append(Task(d))
    return out
