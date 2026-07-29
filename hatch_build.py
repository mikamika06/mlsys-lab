"""Stage the task bank cleanly before it is force-included into the wheel.

`tasks/` sits at the repo root, so it has to be force-included to land inside the
installed package. But `force-include` copies paths verbatim and is NOT subject to
the `exclude` patterns in pyproject.toml — measured: with `exclude` set, 1,596
gitignored `solve.*` scratch files still shipped. Since `solve.*` is where the
learner's (or the verifier's) answer goes, publishing the working tree as-is would
eventually ship a solution next to the task it solves.

So the bank is copied to a staging directory first, with the scratch files left
behind, and the staging copy is what gets force-included. The build then depends on
the contents of the repo rather than on whoever last ran the verifier.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SCRATCH = {"solve.py", "solve.cpp", "solve.cu"}


class CleanBankHook(BuildHookInterface):
    PLUGIN_NAME = "clean-bank"

    def initialize(self, version, build_data):
        src = Path(self.root) / "tasks"
        if not src.is_dir():
            return

        self._tmp = tempfile.mkdtemp(prefix="mlsys-bank-")
        dst = Path(self._tmp) / "tasks"
        shutil.copytree(
            src, dst,
            ignore=shutil.ignore_patterns(*SCRATCH, "__pycache__", "*.pyc", ".DS_Store"),
        )

        kept = sum(1 for _ in dst.glob("*/meta.json"))
        left = sum(1 for n in SCRATCH for _ in src.glob(f"*/{n}"))
        self.app.display_info(f"bank staged: {kept} tasks, {left} scratch files excluded")

        build_data.setdefault("force_include", {})[str(dst)] = "mlsys/tasks"

        # Part-2 projects ship the same way and for the same reason. Their scratch is a
        # learner's copy of the skeleton, which never belongs inside the project folder.
        psrc = Path(self.root) / "projects"
        if psrc.is_dir():
            pdst = Path(self._tmp) / "projects"
            shutil.copytree(
                psrc, pdst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".pytest_cache"),
            )
            n = sum(1 for _ in pdst.glob("*/project.json"))
            ms = sum(len(json.loads((p).read_text())["milestones"])
                     for p in pdst.glob("*/project.json"))
            self.app.display_info(f"projects staged: {n} projects, {ms} milestones")
            build_data["force_include"][str(pdst)] = "mlsys/projects"

    def finalize(self, version, build_data, artifact_path):
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
