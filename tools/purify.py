#!/usr/bin/env python3
"""Rewrite reference solutions by hand, using the local Gemini gateway.

A reference that answers "1-NN from scratch" with `np.argmin` teaches the name of a
function. This asks a model to write the mechanism out — the loop, the accumulator,
the comparison — and only keeps the result if it still clears the task's own gates.

Nothing is trusted: the candidate is graded before it is written, the file is
restored on any failure, and a task that ends up failing verification is reverted
with git rather than left half-converted.

    python3 tools/purify.py --limit 5            # try five
    python3 tools/purify.py --area algorithms    # one area
    python3 tools/purify.py --all -j8            # the whole queue

Transport notes, measured rather than assumed:
  * the gateway returns rendered text, so code must come back inside a ``` fence:
    without one, indentation is stripped and `**` is eaten as markdown bold;
  * the reply carries a UI label ("Gemini said") and sometimes a stray language
    line before the code;
  * 3-5-flash-lite corrupts code (invented identifiers), so the ladder starts at
    3-6-flash and escalates to 3-1-pro.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYFILE = os.path.expanduser("~/.config/gemini-account-gateway/api.key")
BASE = "http://127.0.0.1:8787"
LADDER = ["3-6-flash", "3-1-pro"]
LOG = os.path.join(ROOT, "tools", "purify_log.jsonl")

_spec = importlib.util.spec_from_file_location("cp", os.path.join(ROOT, "tools", "check_pure.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

_print_lock = threading.Lock()
_log_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(msg, flush=True)


def log(row):
    with _log_lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key():
    with open(KEYFILE) as f:
        return f.read().strip()


class RateLimited(Exception):
    def __init__(self, retry_after=300):
        self.retry_after = retry_after
        super().__init__(f"rate limited, retry after {retry_after}s")


def ask(model: str, prompt: str, timeout: int = 300) -> str:
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        if "rate_limited" in raw or e.code == 429:
            try:
                after = json.loads(raw).get("error", {}).get("retry_after", 300)
            except Exception:  # noqa: BLE001
                after = 300
            raise RateLimited(after) from None
        raise RuntimeError(f"HTTP {e.code}: {raw[:200]}") from None
    return data["choices"][0]["message"]["content"]


UI_LABEL = re.compile(r"^\s*(Повідомлення Gemini|Gemini said|Gemini сказал[а]?)\s*", re.I)
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
LANG_LINE = re.compile(r"^\s*(Python|python|py)\s*\n")


def extract(reply: str) -> str:
    """The code, out of a rendered chat message."""
    s = UI_LABEL.sub("", reply)
    m = FENCE.search(s)
    if m:
        s = m.group(1)
    else:
        s = re.sub(r"```[a-z]*", "", s)
    s = LANG_LINE.sub("", s)
    return s.strip() + "\n"


PROMPT = """Rewrite this python file so the computation is written out by hand.

RULES
- No computational numpy. Forbidden: np.exp, np.log, np.sum, np.max, np.min, np.mean,
  np.std, np.dot, np.matmul, np.einsum, np.linalg.*, np.sort, np.argsort, np.argmax,
  np.argmin, np.cumsum, np.clip, np.where, np.round, np.abs, np.sqrt, np.percentile,
  np.histogram, np.unique, np.maximum, np.minimum — and the same as array methods
  (.sum(), .mean(), .max(), .argmax(), .clip(), .round(), .sort(), .any(), .all()).
- Write the loop, the accumulator, the comparison, the rounding. Show the mechanism.
- numpy stays ONLY as plumbing: type annotations, indexing the arrays that arrive as
  input, and building the array you return at the end.
- `math` from the standard library is fine (math.exp, math.sqrt, math.log).
- Keep the EXACT function names, signatures, argument order and return types.
- Keep behaviour identical: same tie-breaking, same rounding, same edge cases, same
  dtype of the returned value. The grader compares against an oracle and will notice.
- Accumulate in the same order the original does, so floating-point results match.
- No comments. A short docstring may stay.

FILE TO REWRITE
{ref}

CONTEXT — how it is graded. Do not change it, do not reproduce it.
{chk}

Reply with the complete new file inside ONE ```python fenced block, and nothing else.
The fence matters: without it the indentation is lost in transit."""


def grade(tid: str, path: str) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, "-m", "mlsys", "grade", tid, "--file", path, "--json"],
                       capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)
    try:
        d = json.loads(r.stdout[r.stdout.index("{"):])
    except Exception:  # noqa: BLE001
        return False, (r.stderr or r.stdout)[-160:].strip()
    return bool(d.get("passed")), json.dumps(d.get("metrics") or {})[:120]


def verify(tid: str) -> tuple[bool, str]:
    r = subprocess.run(["bash", os.path.join(ROOT, "tools", "verify_task.sh"), tid],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    out = (r.stdout + r.stderr).strip().splitlines()
    last = out[-1] if out else ""
    return "TASK_OK" in r.stdout, last[:160]


def one(tid: str, tries: int) -> dict:
    src = os.path.join(ROOT, "tasks", tid, "solution_ref.py")
    chk = os.path.join(ROOT, "tasks", tid, "check.py")
    if not os.path.isfile(src):
        return {"id": tid, "ok": False, "why": "no solution_ref.py"}
    original = open(src, encoding="utf-8").read()
    if not cp.offences(original):
        return {"id": tid, "ok": True, "why": "already by hand", "skipped": True}
    context = open(chk, encoding="utf-8").read()[:3000] if os.path.isfile(chk) else ""

    attempts = []
    for i in range(tries):
        model = LADDER[min(i, len(LADDER) - 1)]
        try:
            reply = ask(model, PROMPT.format(ref=original, chk=context))
        except RateLimited as e:
            attempts.append(f"{model}:rate_limited")
            time.sleep(min(e.retry_after, 60))
            continue
        except Exception as e:  # noqa: BLE001
            attempts.append(f"{model}:{type(e).__name__}")
            continue

        cand = extract(reply)
        try:
            compile(cand, src, "exec")
        except SyntaxError as e:
            attempts.append(f"{model}:syntax L{e.lineno}")
            continue
        left = cp.offences(cand)
        if left:
            attempts.append(f"{model}:still uses {left[0][0]}")
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(cand)
            tmp = f.name
        try:
            passed, metrics = grade(tid, tmp)
        finally:
            os.unlink(tmp)
        if not passed:
            attempts.append(f"{model}:gates {metrics[:60]}")
            continue

        shutil.copy2(src, src + ".bak")
        open(src, "w", encoding="utf-8").write(cand)
        ok, line = verify(tid)
        if ok:
            os.unlink(src + ".bak")
            return {"id": tid, "ok": True, "model": model, "attempt": i + 1,
                    "attempts": attempts, "lines": len(cand.splitlines())}
        shutil.move(src + ".bak", src)
        attempts.append(f"{model}:verify {line[:60]}")

    return {"id": tid, "ok": False, "why": "; ".join(attempts[-3:]) or "no attempt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--area", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-j", type=int, default=8, help="concurrent accounts (max 10)")
    ap.add_argument("--tries", type=int, default=3)
    a = ap.parse_args()

    if a.ids:
        queue = a.ids
    else:
        argv = ["--list"]
        if a.area:
            argv += ["--area", a.area]
        if a.limit:
            argv += ["--limit", str(a.limit)]
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_pure.py")] + argv,
                             capture_output=True, text=True, cwd=ROOT).stdout
        queue = [x for x in out.split() if x]
        if a.limit:
            queue = queue[:a.limit]
    if not queue:
        print("nothing to do")
        return 0
    if not a.all and not a.ids and not a.limit and not a.area:
        print(f"{len(queue)} tasks queued; pass --all to run the lot")
        return 0

    say(f"{len(queue)} tasks · {min(a.j, 10)} at a time · ladder {' then '.join(LADDER)}")
    done = fail = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=min(a.j, 10)) as ex:
        futs = {ex.submit(one, t, a.tries): t for t in queue}
        for fut in cf.as_completed(futs):
            r = fut.result()
            log(dict(r, at=time.strftime("%H:%M:%S")))
            if r["ok"]:
                done += 1
                say(f"  ok   {r['id']:<52} {r.get('model','-')} "
                    f"try {r.get('attempt','-')}  {r.get('lines','')} lines")
            else:
                fail += 1
                say(f"  FAIL {r['id']:<52} {r.get('why','')[:80]}")
    mins = (time.time() - t0) / 60
    say(f"\n{done} rewritten, {fail} left alone, {mins:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
