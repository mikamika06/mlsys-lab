#!/usr/bin/env python3
"""Run the worked examples in every task statement against that task's own reference.

A statement's `## Example` block is the first thing a learner copies. If it disagrees with
the reference, they implement the wrong thing and fail a task they understood correctly.

Nothing caught this before: verify_task.sh checks that the reference passes the gates and the
shipped starter fails them, which says nothing about the prose. One task claimed
`attention_flops(2, 8, 1024, 1024, 64, False) == 2147483648` while its own reference returned
4294967296 — the example counted one of the two matmuls the statement itself describes.

Only examples in the recognised shape are checked:

    result = some_call(args)
    # result == 12345

which is what the generated statements use. Anything else is skipped and counted, so the
coverage of this check is visible rather than assumed.

    python3 tools/check_examples.py
    python3 tools/check_examples.py rwb-attention-flop-count
"""
from __future__ import annotations

import importlib.util
import io
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `x = f(1, 2)` on one line, then `# x == 42` on a later line in the same block.
CALL = re.compile(r"^\s*(\w+)\s*=\s*(\w+)\((.*?)\)\s*$", re.M)
CLAIM = re.compile(r"^\s*#\s*(\w+)\s*==\s*([-\d][\w.\-+e]*)\s*$", re.M)
EXAMPLE = re.compile(r"^## Example\s*\n(.*?)(?=\n## |\Z)", re.S | re.M)
PYBLOCK = re.compile(r"```python\n(.*?)\n```", re.S)


def _example_python(text: str) -> list[str]:
    m = EXAMPLE.search(text)
    return PYBLOCK.findall(m.group(1)) if m else []


# An expected line may carry an explanatory comment — `12288   # (4-1)*1024*4` — which
# doctest compares literally and reports as a mismatch against a bare `12288`. That is the
# checker being wrong, not the task, so the comment is stripped before comparing.
TRAILING_COMMENT = re.compile(r"^(?P<val>[^#\n]*?)\s+#.*$", re.M)
# Statements sometimes gesture at an import that does not exist; that is a style question,
# not a wrong number.
PLACEHOLDER = re.compile(r"\bfrom\s+(your_module|your_solution|solution|module)\s+import\b")


def _strip_expected_comments(block: str) -> str:
    """Remove trailing comments from expected-output lines only, never from >>> lines."""
    out = []
    for line in block.splitlines():
        st = line.strip()
        if st.startswith(">>>") or st.startswith("..."):
            out.append(line)
        else:
            out.append(TRAILING_COMMENT.sub(lambda m: m.group("val").rstrip(), line))
    return "\n".join(out)


def _run_doctest(block: str, mod) -> list[str]:
    """A `>>> call()` example is a promise about the reference's own output."""
    import doctest
    if PLACEHOLDER.search(block):
        return []
    block = _strip_expected_comments(block)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    glb = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")}
    test = doctest.DocTestParser().get_doctest(block, glb, "example", None, 0)
    out = io.StringIO()
    runner.run(test, out=out.write)
    if not runner.failures:
        return []

    # doctest compares text, so a pretty-printed literal — a dict laid out over several
    # lines — reads as a mismatch against the one-line repr Python actually produces. If
    # both sides parse as Python literals and are equal, the page is right and only its
    # formatting differs. Anything that does not parse is reported as before.
    report = out.getvalue()
    exp = re.search(r"Expected:\n(.*?)\nGot:\n(.*?)(?:\n\*{10}|\Z)", report, re.S)
    if exp:
        import ast
        try:
            if ast.literal_eval(exp.group(1).strip()) == ast.literal_eval(exp.group(2).strip()):
                return []
        except Exception:
            pass
    lines = report.strip().splitlines()
    return ["doctest example disagrees with the reference: "
            + " | ".join(l.strip() for l in lines[-4:] if l.strip())[:200]]


def _run_asserts(block: str, mod) -> list[str]:
    """A block that asserts is claiming something; run it and let the assert speak."""
    glb = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("__")}
    glb["__name__"] = "example"
    try:
        exec(compile(block, "<example>", "exec"), glb)
    except AssertionError as e:
        return [f"assertion in the worked example failed: {e or '(no message)'}"]
    except Exception:
        return []          # missing imports or setup the statement only gestures at
    return []


def load(path: Path):
    spec = importlib.util.spec_from_file_location("ref_" + path.parent.name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(task: Path) -> tuple[list[str], int]:
    """Returns (problems, number of claims actually checked)."""
    md = task / "task.md"
    ref = task / "solution_ref.py"
    if not md.is_file() or not ref.is_file():
        return [], 0

    text = md.read_text(encoding="utf-8")
    claims = {m.group(1): m.group(2) for m in CLAIM.finditer(text)}
    calls = {m.group(1): (m.group(2), m.group(3)) for m in CALL.finditer(text)}
    pairs = [(v, calls[v], claims[v]) for v in claims if v in calls]

    blocks = _example_python(text)
    doctests = [b for b in blocks if ">>>" in b]
    asserts = [b for b in blocks if ">>>" not in b and re.search(r"^\s*assert\b", b, re.M)]
    if not pairs and not doctests and not asserts:
        return [], 0

    try:
        mod = load(ref)
    except Exception as e:                       # a broken reference is verify_task's problem
        return [f"reference will not import: {type(e).__name__}: {e}"], 0

    problems: list[str] = []
    checked = 0
    for b in doctests:
        problems += _run_doctest(b, mod)
        checked += 1
    for b in asserts:
        problems += _run_asserts(b, mod)
        checked += 1
    for var, (fn_name, arglist), claimed in pairs:
        fn = getattr(mod, fn_name, None)
        if not callable(fn):
            continue                             # the example calls something else; not ours to judge
        try:
            args = eval(f"({arglist},)", {"__builtins__": {}}, {})   # literals only
        except Exception:
            continue                             # non-literal arguments, cannot replay
        try:
            got = fn(*args)
        except Exception as e:
            problems.append(f"{fn_name}({arglist}) raised {type(e).__name__}: {e}")
            continue
        checked += 1
        try:
            want = float(claimed)
        except ValueError:
            continue
        try:
            gotf = float(got)
        except (TypeError, ValueError):
            continue
        if not (gotf == want or (want and math.isclose(gotf, want, rel_tol=1e-9))):
            problems.append(
                f"statement says {var} == {claimed}, reference returns {got}"
                + (f"  ({gotf / want:g}x)" if want else ""))
    return problems, checked


def main() -> int:
    args = sys.argv[1:]
    tasks = ([ROOT / "tasks" / a for a in args] if args
             else sorted(d for d in (ROOT / "tasks").iterdir() if (d / "meta.json").is_file()))

    bad = 0
    checked_total = 0
    with_claims = 0
    for t in tasks:
        problems, n = check(t)
        checked_total += n
        if n:
            with_claims += 1
        if problems:
            bad += 1
            print(f"FAIL {t.name}")
            for p in problems:
                print(f"       {p}")

    print(f"\n{len(tasks)} tasks, {with_claims} had a replayable worked example, "
          f"{checked_total} claims checked, {bad} disagreed with their reference")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
