#!/usr/bin/env python3
"""Run each concept page's own reproduction snippet and check the page against it.

The whole claim of these pages is that the reader can regenerate the number. If the
snippet does not run, or prints something the page does not say, the page is lying —
and it would be lying in the one way that is worst, because it invites the check.

For every page: extract the `python3 - <<'PY' … PY` block, execute it, and require
that every number the script prints appears somewhere in the page. Numbers are
compared after stripping thousands separators, since tables use them and code does not.

    python3 tools/verify_pages.py
    python3 tools/verify_pages.py concepts/kmeans.md
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEREDOC = re.compile(r"python3 - <<'PY'\n(.*?)\nPY\n", re.S)
PIPLINE = re.compile(r"^pip install (.+)$", re.M)
NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def declared_deps(text: str) -> list[str]:
    """What the page tells the reader to install, beyond this package itself.

    The page is the instruction; if it says `pip install mlsys-lab ml_dtypes` then a
    missing ml_dtypes is the checking environment's problem, not the page's. Report
    it as that, so a real defect is never confused with a missing package.
    """
    out: list[str] = []
    for m in PIPLINE.finditer(text):
        out += [p for p in m.group(1).split()
                if not p.startswith("-") and p not in ("mlsys-lab", "mlsys_lab")]
    return sorted(set(out))


def _sig_digits(s: str) -> int:
    """How many significant digits a written number claims."""
    mantissa = s.lower().split("e")[0].lstrip("-+").replace(".", "").lstrip("0")
    return len(mantissa.rstrip("0")) or 1


def _rounds_to_any(printed: str, candidates: set[str]) -> bool:
    """True when some page value is `printed` rounded to that value's own precision."""
    try:
        pv = float(printed)
    except ValueError:
        return False
    for c in candidates:
        try:
            cv = float(c)
        except ValueError:
            continue
        if cv == pv:
            return True
        digits = _sig_digits(c)
        if digits >= 17:
            continue
        try:
            if float(f"%.{digits - 1}e" % pv) == float(f"%.{digits - 1}e" % cv):
                return True
        except (ValueError, OverflowError):
            continue
    return False


def page_numbers(text: str) -> set[str]:
    """Every number written anywhere in the page, normalised."""
    return {n.replace(",", "") for n in NUM.findall(text.replace(",", ""))}


def check(p: Path) -> list[str]:
    text = p.read_text(encoding="utf-8")
    blocks = HEREDOC.findall(text)
    if not blocks:
        return ["no `python3 - <<'PY'` block to run"]

    problems: list[str] = []
    have = page_numbers(text)

    for i, code in enumerate(blocks):
        r = subprocess.run([sys.executable, "-"], input=code, capture_output=True,
                           text=True, cwd=ROOT, timeout=600)
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            last = tail[-1] if tail else "?"
            missing_dep = re.search(r"No module named '([^']+)'", last)
            if missing_dep and missing_dep.group(1) in {d.replace("-", "_")
                                                       for d in declared_deps(text)}:
                problems.append(
                    f"snippet {i + 1} needs `{missing_dep.group(1)}`, which the page does "
                    f"tell the reader to install — install it here too: "
                    f"pip install {' '.join(declared_deps(text))}")
            else:
                problems.append(f"snippet {i + 1} exited {r.returncode}: {last}")
            continue
        printed = {n.replace(",", "") for n in NUM.findall(r.stdout)}
        # A number the script prints but the page never mentions means the table and
        # the code have drifted apart. Tables legitimately round — 1.110e-15 for a
        # printed 1.1102230246251565e-15 — so a page value counts as a match when it
        # equals the printed value at the page value's own precision. That accepts
        # rounding and still rejects a different number.
        missing = sorted(n for n in printed if n not in have and not _rounds_to_any(n, have))
        # Integers that are obviously loop bookkeeping (0..9) are not claims.
        missing = [n for n in missing if not (n.isdigit() and len(n) <= 1)]
        if missing:
            problems.append(f"snippet {i + 1} printed {len(missing)} value(s) absent from the page: "
                            + ", ".join(missing[:8]))
    return problems


def main() -> int:
    args = [Path(a) for a in sys.argv[1:]]
    files = args or sorted((ROOT / "concepts").glob("*.md"))
    files = [f for f in files if f.name != "README.md"]

    failed = 0
    for f in files:
        bad = check(f)
        if bad:
            failed += 1
            print(f"FAIL {f.name}")
            for b in bad:
                print(f"       {b}")
        else:
            print(f"ok   {f.name}  (snippet ran, every printed number appears in the page)")
    print(f"\n{len(files) - failed}/{len(files)} pages reproduce")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
