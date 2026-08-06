#!/usr/bin/env python3
"""Take numpy out of the tasks that never needed it.

The bank teaches mechanism, and the references were already rewritten by hand —
loops and `math`, not `np.exp`. What stayed behind was the container: a task
about the log-sum-exp identity still opened with `import numpy as np` and typed
its argument `np.ndarray`, so the first thing a learner saw was a dependency the
subject does not have.

Three groups, measured rather than guessed (`--survey`):

  * numpy is the subject — float16 and bfloat16 behaviour, strides, views, bit
    packing, `linalg`. Removing it would delete the task. Left alone.
  * numpy never appears. Nothing to do.
  * numpy is a container: a list of floats in an array's clothing. Those are
    rewritten here, statement and example included, because the statement is
    what the learner copies.

Nothing is trusted. A rewrite is kept only when the task still verifies —
reference passes, shipped starter fails, grade stable across two runs — its
worked example still agrees with its own reference, and no numpy survives in
any file the learner reads. Anything else is reverted with git.

    python3 tools/denumpy.py --survey
    python3 tools/denumpy.py --limit 3
    python3 tools/denumpy.py --all -j4
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "tools", "denumpy_log.jsonl")
LADDER = ["3-6-flash", "3-1-pro"]
VERIFY_TIMEOUT = 180

_spec = importlib.util.spec_from_file_location(
    "bu", os.path.join(ROOT, "tools", "build_unit.py"))
bu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bu)

# numpy as the subject of the task, not as the bag the numbers came in.
SUBJECT = re.compile(
    r"float16|bfloat16|ml_dtypes|\bfp16\b|\bbf16\b|as_strided|\.strides|itemsize|"
    r"frombuffer|\.view\(\s*np|memmap|np\.einsum|finfo|iinfo|int8|uint8|\bqint|"
    r"packbits|unpackbits|byteswap|ascontiguous|\.tobytes\(|np\.linalg")
NUMPY = re.compile(r"\bnumpy\b|\bnp\s*\.|\bnp\b\s*=")
# Some tasks exist to make the learner vectorise. Their gate counts executed
# Python lines, or their statement forbids loops outright, and rewriting them
# into loops inverts exactly what they teach: one such reference came back with
# op_count=22278 against a gate of 50. numpy is the subject there, not the bag.
# The gate that counts executed Python lines, or a statement that forbids loops
# in words. The bare word "vectorize" is not the signal: it appears in prose all
# over the bank, and excluding on it costs real coverage.
VECTOR_GATE = re.compile(
    r"op_count|settrace|setprofile|f_lineno|line execution|"
    r"vectori[sz]ation gate|python_lines|line_events")
VECTOR_STMT = re.compile(
    r"must be vectori[sz]ed|do not use python loops?|"
    r"without (?:a )?python loop|no python loops?", re.I)

LEARNER_FILES = ("starter.py", "solution_ref.py", "task.md")
ALL_FILES = ("task.md", "starter.py", "solution_ref.py", "check.py")
# task.md never goes through the gateway. The reply comes back rendered, which
# closes no fences and turns $\mathbb{R}^{n}$ into a lone R — measured on the
# first pilot, where a statement lost its code fence and its mathematics in one
# turn. The prose is rewritten here instead, by substitution, and checked.
# starter.py is not asked for either. It is two lines, and the gateway strips
# the indentation of the first line of a function body often enough that a
# two-line file fails every time. It is derivable anyway: the signature comes
# from the rewritten reference and the docstring from the starter already there.
MODEL_FILES = ("solution_ref.py", "check.py")

_print_lock = threading.Lock()
_log_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(msg, flush=True)


def log(row):
    with _log_lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read(tid, name):
    p = os.path.join(ROOT, "tasks", tid, name)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read()


def classify(tid):
    files = {n: read(tid, n) for n in ALL_FILES}
    if files["solution_ref.py"] is None:
        return "no reference", files
    # Judged on what the learner reads. check.py is allowed to keep numpy in its
    # oracle, so counting it here left finished tasks in the queue for ever and
    # would have had a restart redo work that was already done.
    learner = "\n".join(files[n] for n in LEARNER_FILES if files.get(n))
    blob = "\n".join(v for v in files.values() if v)
    if not NUMPY.search(learner):
        return "clean", files
    if (SUBJECT.search(blob)
            or VECTOR_GATE.search(files.get("check.py") or "")
            or VECTOR_STMT.search(files.get("task.md") or "")):
        return "subject", files
    return "container", files


def output_is_regenerable(md):
    """Can the shown output be replaced by running the example, structurally.

    Cheap check, no execution: there has to be an Example section with a python
    fence, and either a labelled output fence or exactly one print whose result
    is written beside it.
    """
    head = re.search(r"^#+\s*Example", md or "", re.M | re.I)
    if not head:
        return False
    start = head.start()
    m = EXAMPLE.search(md, start)
    if not m:
        return False
    if OUTPUT_FENCE.search(md, start):
        return True
    return m.group(1).count("print(") == 1


def queue(include_printed=False):
    """Tasks worth spending a turn on.

    A statement that shows an array printed the numpy way is rejected by a gate
    that only fires after the model has been paid four times. Those are held
    back here instead: rewriting them honestly means running the reference to
    get the real output, which is a separate pass.
    """
    out, deferred = [], []
    for d in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", ""))):
        tid = os.path.basename(d.rstrip("/"))
        kind, files = classify(tid)
        if kind != "container":
            continue
        md = files.get("task.md") or ""
        if not include_printed and NPPRINT.search(md) and not output_is_regenerable(md):
            deferred.append(tid)
            continue
        out.append(tid)
    if deferred:
        with open(os.path.join(ROOT, "tools", "denumpy_deferred.txt"), "w") as f:
            f.write("\n".join(deferred) + "\n")
    return out


PROMPT = """Rewrite one exercise so it uses no numpy at all.

numpy is not the subject here. It is being used as a container for numbers that
a plain Python list holds just as well, and it puts a dependency in front of a
learner who is meant to be looking at the mechanism.

Rules, all of them load-bearing:

1. No numpy anywhere in task.md, starter.py or solution_ref.py. No import, no
   `np.` call, no `np.ndarray` annotation, not in prose, not in the example.
2. Arguments and return values become plain Python: `list[float]`,
   `list[list[float]]`, `int`, `float`, `bool`. The function name, the parameter
   names, their order and their default values stay exactly as they are — only
   the annotations change. Learners already have this signature, and a rewrite
   that renames a parameter or drops a default breaks their code.
3. solution_ref.py stays written out by hand: explicit loops, an accumulator,
   `math.exp` / `math.log` where a scalar function is needed. It already is —
   do not turn it back into library calls, and do not change what it computes.
4. task.md and starter.py are not yours to write and are not in the reply. They
   are regenerated from the reference you send, so the signature you choose is
   the signature the learner will see. Choose it carefully.
5. (unused): update the signature block, the Input and Output prose (a "1-D
   NumPy array of float64" is now "a list of floats"), and the `## Example`
   block. The example has to be runnable and its stated result has to be what
   the new reference returns. Do not touch the explanation, the mathematics or
   the title.
6. check.py: the grader calls the learner's function with plain lists and
   compares against its own oracle. The oracle may keep numpy if that is what
   keeps it correct, but everything crossing the boundary to the solution is a
   list. The gate, its metric names and its tolerances stay exactly as they are.
7. Numerical behaviour does not change. The same inputs give the same outputs to
   the last bit where that was true before.

Reply with the changed files, nothing else. Every file as a line

FILE: <name>

followed by its complete new contents. No commentary before or after.

=== the function is named `{entry}` and must keep that exact name

=== task id: {tid}

{files}
"""


def contract(tid, files):
    parts = []
    for name in ("task.md",) + MODEL_FILES:
        if files.get(name) is None:
            continue
        parts.append("FILE: %s\n```\n%s\n```" % (name, files[name]))
    return PROMPT.format(tid=tid, entry=entry_name(files.get("starter.py")) or "?",
                         files="\n\n".join(parts))


def verify(tid):
    r = subprocess.run(["bash", os.path.join(ROOT, "tools", "verify_task.sh"), tid],
                       capture_output=True, text=True, cwd=ROOT,
                       timeout=VERIFY_TIMEOUT)
    line = (r.stdout.strip().splitlines() or [""])[-1]
    return line.startswith("TASK_OK"), line[:150]


def examples_ok(tid):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_examples.py"), tid],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    out = (r.stdout + r.stderr).strip()
    if "FAIL" in out or "MISMATCH" in out:
        return False, out.splitlines()[-1][:120] if out else "example failed"
    return True, ""


def numpy_left(tid):
    for name in LEARNER_FILES:
        body = read(tid, name)
        if body and NUMPY.search(body):
            return name
    return None


def revert(tid):
    subprocess.run(["git", "checkout", "--", os.path.join("tasks", tid)],
                   cwd=ROOT, capture_output=True)


def syntax_error(files):
    for name, body in files.items():
        if not name.endswith(".py"):
            continue
        if not body.strip():
            return "%s is empty" % name
        try:
            ast.parse(body)
        except SyntaxError as e:
            return "%s: %s line %s" % (name, type(e).__name__, e.lineno)
    return None


CTOR = [
    (re.compile(r"np\.zeros\(\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(?:,[^)]*)?\)"),
     r"[[0.0] * \2 for _ in range(\1)]"),
    (re.compile(r"np\.ones\(\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(?:,[^)]*)?\)"),
     r"[[1.0] * \2 for _ in range(\1)]"),
    (re.compile(r"np\.full\(\s*(\w+)\s*,\s*([^,)]+?)\s*(?:,[^)]*)?\)"),
     r"[\2] * \1"),
    (re.compile(r"np\.eye\(\s*(\w+)\s*(?:,[^)]*)?\)"),
     r"[[1.0 if i == j else 0.0 for j in range(\1)] for i in range(\1)]"),
    (re.compile(r"np\.(?:array|asarray)\(\s*(\[.*?\])\s*(?:,\s*dtype\s*=\s*[\w.]+\s*)?\)", re.S), r"\1"),
    (re.compile(r"np\.zeros\(\s*(\w+)\s*(?:,[^)]*)?\)"), r"[0.0] * \1"),
    (re.compile(r"np\.ones\(\s*(\w+)\s*(?:,[^)]*)?\)"), r"[1.0] * \1"),
    (re.compile(r"np\.arange\(\s*(\w+)\s*(?:,[^)]*)?\)"), r"list(range(\1))"),
]

PROSE = [
    (re.compile(r"The implementation may use any NumPy operations;\s*", re.I),
     "The implementation uses plain Python; "),
    (re.compile(r"\b(?:one|1)[\u2010-\u2015-]?dimensional\s+NumPy\s+arrays?\b", re.I),
     "list of floats"),
    (re.compile(r"\b(?:two|2)[\u2010-\u2015-]?dimensional\s+NumPy\s+arrays?\b", re.I),
     "list of lists of floats"),
    (re.compile(r"\b1-?D\s+NumPy\s+arrays?\b", re.I), "list of floats"),
    (re.compile(r"\b2-?D\s+NumPy\s+arrays?\b", re.I), "list of lists of floats"),
    (re.compile(r"\bNumPy\s+arrays?\b", re.I), "list"),
    (re.compile(r"\bNumPy\b"), "Python"),
    (re.compile(r"`?\bnp\.ndarray\b`?"), "list[float]"),
    (re.compile(r"`?\bnp\.float64\b`?"), "float"),
    (re.compile(r"`?\bnp\.float32\b`?"), "float"),
    (re.compile(r"`?\bnp\.int(?:64|32)\b`?"), "int"),
    (re.compile(r"`?\bnp\.bool_\b`?"), "bool"),
]

LEFTOVER = re.compile(r"\bnumpy\b|\bnp\s*\.", re.I)
# numpy prints an array as numbers separated by spaces. A list does not, so a
# statement showing that output is now describing something that will not happen.
NPPRINT = re.compile(r"\[\s*[-\d][\d\.\se\+\-]*\s+[-\d][\d\.\se\+\-]*\]")


def _replace_signature(text, signature):
    """Swap the def line in the statement for the one the code actually has.

    The statement writes its signature across several lines as often as not, so
    the end of it is found by matching parentheses rather than by looking for a
    newline.
    """
    if not signature:
        return text
    name = signature.split("(")[0].split()[-1]
    m = re.search(r"^[ \t]*def[ \t]+" + re.escape(name) + r"[ \t]*\(", text, re.M)
    if not m:
        return text
    i = m.end() - 1
    depth = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if i >= len(text):
        return text
    j = text.find(":", i)
    if j < 0:
        return text
    return text[:m.start()] + signature + text[j:]


def rewrite_statement(text, signature):
    """Take numpy out of a statement without disturbing anything else.

    Substitution, not regeneration: the fences, the LaTeX and the explanation
    survive untouched, because none of them are about numpy. Whatever this does
    not recognise is left in place and the caller rejects the task, so a
    statement is never shipped half-converted.
    """
    text = _replace_signature(text, signature)
    kept = []
    for line in text.split("\n"):
        if re.match(r"\s*(?:import\s+numpy|from\s+numpy)\b", line):
            continue
        kept.append(line)
    text = "\n".join(kept)
    for pat, rep in CTOR:
        text = pat.sub(rep, text)
    for pat, rep in PROSE:
        text = pat.sub(rep, text)
    return text


def entry_name(source):
    """The name of the function the task is about.

    The first def in a file is not it: a rewritten reference often opens with a
    helper — _softmax, _matmul — and taking that one made the signature check
    report a rename that never happened and built a starter for the wrong
    function.
    """
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return None
    names = [n.name for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    public = [n for n in names if not n.startswith("_")]
    return (public or names or [None])[0]


def _func(source, want=None):
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return None
    funcs = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if want:
        for n in funcs:
            if n.name == want:
                return n
    public = [n for n in funcs if not n.name.startswith("_")]
    return (public or funcs or [None])[0]


def params_of(source, want=None):
    """(name, [(param, default_repr)]) of the task's function."""
    try:
        ast.parse(source)
    except SyntaxError:
        return None, []
    for node in [_func(source, want)] if _func(source, want) else []:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
            defaults = [ast.unparse(d) for d in args.defaults if d is not None]
            defaults += [ast.unparse(d) for d in args.kw_defaults if d is not None]
            return node.name, list(zip(names[len(names) - len(defaults):], defaults))
    return None, []


def signature_drift(old_starter, new_reference):
    """The rewrite must not change the interface the learner already has."""
    want = entry_name(old_starter)
    on, od = params_of(old_starter or "", want)
    nn, nd = params_of(new_reference or "", want)
    if on and nn and on != nn:
        return "function renamed %s -> %s" % (on, nn)
    o_names = [a.arg for a in _arglist(old_starter, want)]
    n_names = [a.arg for a in _arglist(new_reference, want)]
    if o_names and n_names and o_names != n_names:
        return "parameters changed %s -> %s" % (o_names, n_names)
    if dict(od) != dict(nd):
        return "defaults changed %s -> %s" % (dict(od), dict(nd))
    return None


def _arglist(source, want=None):
    node = _func(source, want)
    if node is None:
        return []
    a = node.args
    return a.posonlyargs + a.args + a.kwonlyargs


def signature_of(source, want=None):
    """The def line, rebuilt from the syntax tree.

    Reading it back out of the source with a regular expression missed the
    multi-line declarations these references like to use, and a missing
    signature is reported as "no signature in reference" — a rewrite thrown
    away over the way it was formatted.
    """
    node = _func(source, want)
    if node is None:
        return ""
    try:
        args = ast.unparse(node.args)
        ret = " -> " + ast.unparse(node.returns) if node.returns is not None else ""
    except Exception:  # noqa: BLE001
        return ""
    return "def %s(%s)%s" % (node.name, args, ret)


LINES_PROMPT = """Rewrite these lines so they do not mention numpy.

They come from an exercise statement whose code now uses plain Python lists
instead of numpy arrays. Each line has to keep saying what it says: same claim,
same formula, same example, same LaTeX, only expressed without numpy. An array
constructor becomes a list literal, `np.sum(...)` becomes the plain Python that
computes the same thing, "NumPy array" becomes "list".

The function signature is now:

{signature}

Reply with exactly one line per input line, in the same order, each prefixed by
its number and a tab:

1\t<rewritten line>
2\t<rewritten line>

No commentary, no extra lines, no code fences. If a line needs no change, send
it back unchanged.

=== lines
{lines}
"""


def _numbered(reply, count):
    """Split a numbered reply that may have lost its line breaks.

    The separators come back as spaces often enough that the answers arrive as
    `1 first line 2 second line` on one physical line. Splitting on newlines
    took the first item and dropped the rest, so the whole rewrite was thrown
    away. Markers are found in order instead: only a `2` that follows item 1
    starts item 2, which a stray number in the prose cannot fake.
    """
    out = {}
    pos = 0
    for n in range(1, count + 1):
        m = re.compile(r"(?:^|\s)%d(?:\t|\.\s|\)\s|\s)" % n).search(reply, pos)
        if not m:
            return out
        start = m.end()
        nxt = re.compile(r"(?:^|\s)%d(?:\t|\.\s|\)\s|\s)" % (n + 1)).search(reply, start)
        end = nxt.start() if nxt else len(reply)
        out[n] = reply[start:end].strip("\n").rstrip()
        pos = end
    return out


def rewrite_numpy_lines(after, signature, ask_one):
    """Rewrite only the lines that still mention numpy, and splice them back.

    Sending the whole statement to be rewritten cost a formula: the model
    returned clean prose and silently dropped a display-math block from the
    section about the gate. Handing it only the offending lines makes the rest
    of the statement untouchable by construction, and it is a much smaller
    prompt.
    """
    lines = after.split("\n")
    idx = [i for i, ln in enumerate(lines) if LEFTOVER.search(ln)]
    if not idx:
        return after
    numbered = "\n".join("%d\t%s" % (n + 1, lines[i]) for n, i in enumerate(idx))
    try:
        reply = ask_one(LINES_PROMPT.format(signature=signature, lines=numbered))
    except Exception:  # noqa: BLE001
        return after
    got = _numbered(reply, len(idx))
    if len(got) != len(idx):
        return after
    out = list(lines)
    for n, i in enumerate(idx):
        repl = got.get(n + 1)
        if repl is None or LEFTOVER.search(repl):
            return after
        out[i] = repl
    return "\n".join(out)


EXAMPLE = re.compile(r"```python\n(.*?)```", re.S)
OUTPUT_FENCE = re.compile(r"(\n(?:Output|Result)[^\n]*\n+```[a-z]*\n)(.*?)(```)", re.S | re.I)


def refresh_output(md, reference_src, tid):
    """Replace the illustrative output with what the example actually prints.

    These statements show an array the way numpy prints one. After the rewrite
    the code returns a list, so the block is no longer describing anything that
    happens. Running it is the only honest way to say what it prints now.
    """
    # The first python fence in a statement is the signature block, not the
    # example; searching from the top runs `def f(...): ...` and gets an
    # IndentationError instead of an output.
    head = re.search(r"^#+\s*Example", md, re.M | re.I)
    start = head.start() if head else 0
    m = EXAMPLE.search(md, start)
    om = OUTPUT_FENCE.search(md, start)
    if not m or not om:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        rp = os.path.join(tmp, "solution_ref.py")
        ep = os.path.join(tmp, "example.py")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(reference_src)
        with open(ep, "w", encoding="utf-8") as f:
            f.write(m.group(1))
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "tools", "run_example.py"), rp, ep],
                capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return None
    if "OK" not in r.stderr or not r.stdout.strip():
        return None
    return md[:om.start(2)] + r.stdout.rstrip() + "\n" + md[om.end(2):]


WHY = []


STDLIB_OK = {"math", "random", "itertools", "collections", "functools", "heapq",
             "bisect", "struct", "json", "sys", "os", "typing", "numpy", "time"}


def _example_modules(code):
    """Module names an example imports that only the reference can satisfy.

    Statements import the solution under whatever name reads well —
    `from your_module import layer_norm`, `from pairwise_l2_matrix import ...` —
    so the reference has to be importable under each of them.
    """
    names = {"solution_ref", "solution"}
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
                         code, re.M):
        name = (m.group(1) or m.group(2) or "").split(".")[0]
        if name and name not in STDLIB_OK:
            names.add(name)
    return names


def _run_example(reference_src, code):
    with tempfile.TemporaryDirectory() as tmp:
        rp = os.path.join(tmp, "solution_ref.py")
        ep = os.path.join(tmp, "example.py")
        for name in _example_modules(code):
            with open(os.path.join(tmp, name + ".py"), "w", encoding="utf-8") as f:
                f.write(reference_src)
        with open(rp, "w", encoding="utf-8") as f:
            f.write(reference_src)
        with open(ep, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "tools", "run_example.py"), rp, ep],
                capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return None
    if "OK" not in r.stderr:
        WHY.append("run: " + (r.stderr.strip().splitlines() or ["?"])[-1][:90])
        return None
    return r.stdout


def refresh_inline_output(md, reference_src, tid):
    """Rewrite the output a statement shows as a comment beside its print.

    Most of these statements do not label the output at all: they write
    `print(preds)  # [1 1]`, or follow the print with a block of `#` lines. Once
    the code returns lists those comments describe an array that no longer
    exists, and the only honest replacement is what the example now prints.
    Handled for a single print, which is the shape they all use; anything else
    is left alone rather than guessed at.
    """
    head = re.search(r"^#+\s*Example", md, re.M | re.I)
    start = head.start() if head else 0
    m = EXAMPLE.search(md, start)
    if not m:
        return None
    code = m.group(1)
    if code.count("print(") != 1:
        return None
    out = _run_example(reference_src, code)
    if not out or not out.strip():
        return None
    lines = out.rstrip("\n").split("\n")

    src = code.split("\n")
    idx = next((i for i, ln in enumerate(src) if "print(" in ln), None)
    if idx is None:
        return None
    src[idx] = re.sub(r"\s*#.*$", "", src[idx]).rstrip()
    tail = idx + 1
    while tail < len(src) and src[tail].lstrip().startswith("#"):
        tail += 1
    comment = ["# " + ln if ln else "#" for ln in lines]
    if len(lines) == 1:
        src[idx] = src[idx] + "  " + comment[0]
        new = src[:idx + 1] + src[tail:]
    else:
        new = src[:idx + 1] + comment + src[tail:]
    return md[:m.start(1)] + "\n".join(new) + md[m.end(1):]


def statement_intact(before, after):
    """Fences balanced and the mathematics still there."""
    if before.count("```") != after.count("```"):
        return "fence count changed"
    if after.count("```") % 2:
        return "unbalanced fences"
    if before.count("$") and after.count("$") < before.count("$"):
        return "lost %d math delimiters" % (before.count("$") - after.count("$"))
    if len(after) < len(before) * 0.6:
        return "statement lost %d%% of its length" % (100 - 100 * len(after) // len(before))
    return None


def rebuild_starter(old_starter, reference):
    """The starter that goes with a reference: same signature, no body."""
    sig = signature_of(reference, entry_name(old_starter))
    if not sig:
        return None
    doc = ""
    m = re.search(r"^def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n(\s+(?:\"\"\"|\'\'\')(?:.|\n)*?(?:\"\"\"|\'\'\'))",
                  old_starter or "", re.M)
    if m:
        doc = m.group(1).rstrip() + "\n"
    head = ""
    for line in (reference or "").split("\n"):
        if line.startswith("def "):
            break
        if re.match(r"\s*(import|from)\s+(math|itertools|collections|functools|heapq|bisect|struct|random)\b", line):
            head += line + "\n"
    body = doc + "    raise NotImplementedError('your code here')\n"
    return (head + "\n" if head else "") + sig + ":\n" + body


def write_files(tid, files):
    written = []
    for name, body in files.items():
        base = os.path.basename(name)
        if base not in ALL_FILES:
            continue
        p = os.path.join(ROOT, "tasks", tid, base)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body.rstrip() + "\n")
        written.append(base)
    return written


def one(tid, turns):
    kind, files = classify(tid)
    if kind != "container":
        return {"id": tid, "ok": True, "skipped": True, "why": kind}
    history = []
    nudged = False
    prompt = contract(tid, files)
    conv = "mlsys-denumpy-" + tid
    for turn in range(turns):
        model = LADDER[min(turn // 2, len(LADDER) - 1)]
        try:
            reply = bu.ask(model, prompt, conv)
        except Exception as e:  # noqa: BLE001
            history.append("%s:%s" % (model, str(e)[:60]))
            continue
        got = bu.parse_files(reply)
        got = {os.path.basename(k): v for k, v in got.items()
               if os.path.basename(k) in MODEL_FILES}
        broken = syntax_error(got)
        if broken:
            # Writing a file that cannot be parsed spends a verification run to
            # learn what ast.parse knows for free. The damage is intermittent and
            # confined to one file, so the chat is kept and only that file is
            # asked for again — resending the whole contract costs four files to
            # repair one.
            history.append("%s:%s" % (model, broken))
            name = broken.split(":")[0]
            prompt = ("%s did not parse: %s. Send that one file again, complete, "
                      "as `FILE: %s` followed by a fenced block. Nothing else."
                      % (name, broken, name))
            continue
        if not got:
            # The contract is already in this chat. A nudge costs two lines; a
            # fresh conversation costs the whole contract again, so that is the
            # second answer to an empty reply, not the first.
            history.append("%s:no files" % model)
            if nudged:
                conv = "mlsys-denumpy-%s-r%d" % (tid, turn)
                prompt = contract(tid, files)
                nudged = False
            else:
                nudged = True
                prompt = ("Send the files now: solution_ref.py and check.py, each "
                          "as `FILE: <name>` followed by a fenced block. "
                          "No commentary.")
            continue

        drift = signature_drift(files.get("starter.py"), got.get("solution_ref.py"))
        if drift:
            history.append("%s:%s" % (model, drift))
            prompt = ("The signature changed: %s. Send solution_ref.py again "
                      "with the original function name, parameter names, order "
                      "and default values; only the annotations may change."
                      % drift)
            continue
        starter = rebuild_starter(files.get("starter.py"), got.get("solution_ref.py", ""))
        if not starter:
            want = entry_name(files.get("starter.py")) or "the task function"
            history.append("%s:reference does not define %s" % (model, want))
            prompt = ("solution_ref.py does not define `%s`. That is the function "
                      "the task is about and the name cannot change. Send "
                      "solution_ref.py again, defining it at module level."
                      % want)
            continue
        got["starter.py"] = starter
        written = write_files(tid, got)
        # The statement follows the code: its signature block is taken from the
        # starter that was just written, so the two cannot disagree.
        before = files["task.md"] or ""
        after = rewrite_statement(before, signature_of(
            got.get("solution_ref.py", ""), entry_name(files.get("starter.py"))))
        if LEFTOVER.search(after):
            after = rewrite_numpy_lines(
                after, signature_of(got.get("solution_ref.py", "")),
                lambda pr: bu.ask(model, pr, conv + "-lines"))

        broke = statement_intact(before, after)
        if not broke and NPPRINT.search(after):
            WHY.clear()
            fresh = (refresh_output(after, got.get("solution_ref.py", ""), tid)
                     or refresh_inline_output(after, got.get("solution_ref.py", ""), tid))
            if fresh and not NPPRINT.search(fresh):
                after = fresh
                broke = statement_intact(before, after)
            else:
                why = (WHY[-1] if WHY else
                       ("still numpy-shaped after regeneration" if fresh
                        else "no regeneration path matched"))
                history.append("%s:output %s" % (model, why))
        if not broke and NPPRINT.search(after):
            # The statement shows an array printed the way numpy prints one.
            # Rewriting that honestly means running the reference, which is a
            # separate pass; shipping it as is would state an output the task
            # no longer produces.
            broke = "shows numpy-printed output"
        if broke:
            history.append("%s:statement %s" % (model, broke))
            revert(tid)
            continue
        with open(os.path.join(ROOT, "tasks", tid, "task.md"), "w",
                  encoding="utf-8") as f:
            f.write(after)
        left = numpy_left(tid)
        if left:
            history.append("%s:numpy still in %s" % (model, left))
            revert(tid)
            prompt = ("numpy is still present in %s. Remove every trace of it "
                      "from task.md, starter.py and solution_ref.py, and send "
                      "those files again in the same FILE: format." % left)
            continue
        ok, line = verify(tid)
        if not ok:
            history.append("%s:%s" % (model, line[-90:]))
            revert(tid)
            prompt = ("`tools/verify_task.sh %s` reports:\n\n    %s\n\nFix it and "
                      "send the corrected files again. The reference has to pass "
                      "every gate and the starter has to fail, and still no numpy."
                      % (tid, line))
            continue
        ok, why = examples_ok(tid)
        if not ok:
            history.append("%s:example %s" % (model, why[-70:]))
            revert(tid)
            prompt = ("The worked example in task.md disagrees with the "
                      "reference: %s\nSend task.md again with an example whose "
                      "stated result is what the reference returns." % why)
            continue
        return {"id": tid, "ok": True, "model": model, "turn": turn + 1,
                "files": sorted(got), "history": history}
    revert(tid)
    return {"id": tid, "ok": False, "why": "; ".join(history[-3:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--include-printed", action="store_true",
                    help="also take the statements that show numpy-printed output")
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()

    if a.survey:
        counts = {}
        for d in sorted(glob.glob(os.path.join(ROOT, "tasks", "*", ""))):
            tid = os.path.basename(d.rstrip("/"))
            counts[classify(tid)[0]] = counts.get(classify(tid)[0], 0) + 1
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print("%-14s %d" % (k, v))
        return 0

    ids = a.ids or queue(a.include_printed)
    if a.limit:
        ids = ids[:a.limit]
    say("%d tasks · %d at a time · up to %d turns" % (len(ids), a.j, a.turns))

    done = fail = 0
    with cf.ThreadPoolExecutor(max_workers=a.j) as pool:
        futs = {pool.submit(one, t, a.turns): t for t in ids}
        for fut in cf.as_completed(futs):
            tid = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                row = {"id": tid, "ok": False, "why": "%s: %s" % (type(e).__name__, e)}
            log(row)
            if row.get("skipped"):
                continue
            if row["ok"]:
                done += 1
                say("  ok   %-52s %s turn %s" % (tid, row.get("model"), row.get("turn")))
            else:
                fail += 1
                say("  FAIL %-52s %s" % (tid, row.get("why", "")[:70]))
    say("rewritten %d · failed %d" % (done, fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
