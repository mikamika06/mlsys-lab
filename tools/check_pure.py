#!/usr/bin/env python3
"""Which reference solutions still let a library do the work.

The bank teaches mechanism. A reference that answers "dot product from scratch"
with `np.dot` teaches the name of a function, so the rule is that the solution is
written out by hand — the loop, the accumulator, the rounding — and numpy stays on
the grader's side of the fence, where being right matters more than being legible.

Type annotations and array plumbing (`np.ndarray`, `np.asarray`, `np.float32`) are
not the work and do not count. Only the calls that would otherwise *be* the answer.

    python3 tools/check_pure.py            # summary
    python3 tools/check_pure.py --list     # one id per line, for a build queue
    python3 tools/check_pure.py --area gpu-cuda
"""
import argparse
import ast
import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Calls that perform the computation a task is usually asking for. Everything else
# — construction, casting, reshaping, annotation — is plumbing around the answer.
WORK = re.compile(
    r"\bnp\.("
    r"exp|log|log2|log10|log1p|expm1|sum|prod|max|min|amax|amin|mean|median|std|var|"
    r"sort|argsort|argmax|argmin|argpartition|partition|"
    r"dot|matmul|inner|outer|einsum|tensordot|cross|"
    r"linalg|fft|"
    r"cumsum|cumprod|diff|gradient|"
    r"clip|where|round|rint|floor|ceil|trunc|sign|abs|sqrt|cbrt|square|power|"
    r"percentile|quantile|histogram|bincount|unique|searchsorted|"
    r"corrcoef|cov|convolve|correlate|"
    r"maximum|minimum|logaddexp|"
    r"isclose|allclose|"
    r"softmax|tanh|sinh|cosh|sin|cos"
    r")\b")

METHOD = re.compile(
    r"\.(sum|mean|max|min|argmax|argmin|std|var|prod|cumsum|dot|clip|round|sort|"
    r"argsort|any|all)\s*\(")


def code_of(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def offences(src):
    """(call, line) for every place a library does the work."""
    out = []
    try:
        tree = ast.parse(src)
        doc_lines = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    doc_lines.add(ln)
    except SyntaxError:
        doc_lines = set()
    for i, line in enumerate(src.splitlines(), 1):
        if i in doc_lines or line.strip().startswith("#"):
            continue
        for m in WORK.finditer(line):
            out.append((m.group(0), i))
        for m in METHOD.finditer(line):
            # a method call on a name that looks like an array, not on a list
            out.append((m.group(0).strip("("), i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print ids only")
    ap.add_argument("--area", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    curriculum = {}
    cf = os.path.join(ROOT, "src", "mlsys", "task_list2.json")
    if os.path.isfile(cf):
        with open(cf, encoding="utf-8") as f:
            curriculum = {r["id"]: r for r in json.load(f)["rows"]}

    dirty, clean = [], 0
    by_area = collections.Counter()
    for d in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", ""))):
        ref = os.path.join(d, "solution_ref.py")
        if not os.path.isfile(ref):
            continue
        tid = os.path.basename(d.rstrip("/"))
        area = (curriculum.get(tid) or {}).get("area", "?")
        if a.area and a.area not in area:
            continue
        hits = offences(code_of(ref))
        if hits:
            dirty.append((area, tid, hits))
            by_area[area] += 1
        else:
            clean += 1

    if a.list:
        for _, tid, _ in dirty[:a.limit or None]:
            print(tid)
        return 0

    total = len(dirty) + clean
    print(f"{clean}/{total} reference solutions are written by hand")
    print(f"{len(dirty)} still let a library do the work\n")
    for area, n in by_area.most_common():
        print(f"  {area:34} {n}")
    if dirty:
        print("\nworst offenders (fewest lines of their own):")
        ranked = sorted(dirty, key=lambda x: -len(x[2]))[:5]
        for area, tid, hits in ranked:
            calls = collections.Counter(h[0] for h in hits)
            print(f"  {tid:52} {', '.join(f'{k}×{v}' for k, v in calls.most_common(4))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
