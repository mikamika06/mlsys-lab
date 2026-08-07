#!/usr/bin/env python3
"""Build Part-2 units through the local Gemini gateway.

A unit is a directory of files that must agree with each other: a ticket, a
skeleton that fails, a reference that passes, and a checker per milestone. Getting
that right is rarely a single shot, so the build is a conversation — the gateway's
`conversation_key` keeps one chat per unit, and a repair turn says only what broke
instead of restating the whole contract. Without it every retry starts from
amnesia and re-derives the same wrong thing.

    python3 tools/build_unit.py m-foo-bar            # one
    python3 tools/build_unit.py --tier T0 --limit 20 # a slice of the queue
    python3 tools/build_unit.py --all -j8

The contract is machine-checked, not taken on the model's word:
tools/verify_project.py must report `reference N/N, skeleton 0/N`.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(ROOT, "tools", "specs2")
KEYFILE = os.path.expanduser("~/.config/gemini-account-gateway/api.key")
BASE = "http://127.0.0.1:8787"
LADDER = ["3-6-flash", "3-1-pro"]
LOG = os.path.join(ROOT, "tools", "build_unit_log.jsonl")
TEMPLATE_M = "projects/m-build-kv-cache-groups-from-a-hybrid-model-config"
TEMPLATE_L = "projects/p-continuous-batching-scheduler"

RAW_DUMP = []
_lock = threading.Lock()


def say(m):
    with _lock:
        print(m, flush=True)


def log(row):
    with _lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


GATE = threading.Event()
GATE.set()


def _wait_for_gateway(reason):
    """Hold every worker until the gateway answers again.

    A gateway that stops accepting connections fails a unit in milliseconds, so
    nine workers can burn the entire queue between two progress reports — which
    is exactly what happened: 1,217 units "failed" in one outage without a single
    request being answered. A transport error is not a fact about the unit, so it
    must not consume it.
    """
    if not GATE.is_set():
        GATE.wait()
        return
    GATE.clear()
    print(f"gateway unreachable ({reason}); pausing", flush=True)
    delay = 5
    while True:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=15) as r:
                if json.load(r).get("ok"):
                    break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(delay)
        delay = min(delay * 2, 120)
    print("gateway back; resuming", flush=True)
    GATE.set()


def ask(model, prompt, conv_key, timeout=600, tries=4, expect_files=False):
    body = {"model": model, "conversation_key": conv_key,
            "messages": [{"role": "user", "content": prompt}]}
    payload = json.dumps(body).encode()
    last = None
    for attempt in range(tries):
        GATE.wait()
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=payload,
            headers={"Authorization": "Bearer " + open(KEYFILE).read().strip(),
                     "Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            break
        except urllib.error.HTTPError as e:
            code = e.code
            text = e.read().decode("utf-8", "replace")[:160]
            if code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                last = f"HTTP {code}: {text}"
                time.sleep(min(20 * (attempt + 1), 90))
                continue
            raise RuntimeError(f"HTTP {code}: {text}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = f"{type(e).__name__}: {str(e)[:80]}"
            _wait_for_gateway(last)
            continue
    else:
        raise RuntimeError("gateway unreachable: " + str(last))
    # The gateway usually answers with choices, but a refused or errored job comes
    # back shaped differently; reading it blind turned that into KeyError('choices')
    # and cost a whole turn with no explanation.
    ch = d.get("choices")
    if not ch:
        err = d.get("error") or d
        text = json.dumps(err, ensure_ascii=False)[:200]
        # A cold browser pool answers `execution_failed: Chrome ...` for the
        # first jobs after a gateway restart. That is the pool warming up, not
        # anything about this unit, so it is retried like a dropped connection.
        transient = ("execution_failed" in text or "Chrome" in text
                     or "browser" in text.lower())
        if transient and tries > 1:
            time.sleep(min(15 * (5 - tries), 45))
            return ask(model, prompt, conv_key, timeout, tries - 1, expect_files)
        raise RuntimeError("no choices: " + text)
    content = ch[0]["message"]["content"]
    # A reply of forty bytes carrying the model's thinking header and nothing
    # else — "Defining the Exercise\nОтвет Gemini" — is a transport failure, not
    # an answer. Spending a whole turn on it is the expensive way to find out,
    # so when files were expected and none arrived, the request is simply made
    # again.
    # Any reply without a single FILE: marker is unusable when files were asked
    # for, whatever its length: forty bytes of thinking header, or prose about
    # what it intends to do. Retrying costs one request; letting it through
    # costs one of the unit's turns.
    if expect_files and "FILE:" not in content and tries > 1:
        time.sleep(3)
        return ask(model, prompt, conv_key, timeout, tries - 1, expect_files)
    # The other shape of the same failure: the files arrive but every newline
    # inside them is gone, so a checker reads `import ref import numpy as np`
    # and `def check(workdir): from x import y` on two lines. No model writes
    # that; it is the reply losing its structure on the way back.
    if expect_files and "FILE:" in content:
        lines = content.count("\n") or 1
        if len(content) / lines > 150 and tries > 1:
            time.sleep(3)
            return ask(model, prompt, conv_key, timeout, tries - 1, expect_files)
    return content


UI = re.compile(r"^\s*(Повідомлення Gemini|Gemini said|Gemini сказал[а]?|Ответ Gemini|Відповідь Gemini)\s*", re.I)
# The newline after the path is sometimes eaten in transit, so the marker line
# arrives as `FILE: solution_ref.py import math`. Anchoring on end-of-line threw
# that whole reply away; the path is the first token and whatever follows it is
# the first line of the file.
MARK = re.compile(r"^FILE:[ \t]*(\S+)[ \t]*(.*)$", re.M)
# The gateway renders markdown and hands back text, so the ``` fences never survive
# — what arrives is the block's language name on a line of its own. Indentation
# does survive, which is the part that matters, so the file boundary is the FILE:
# marker and the label is simply dropped.
LABEL = re.compile(r"^(?:Python|python|JSON|json|Markdown|markdown|Text|text|Bash|bash|py)\s*$")


def parse_files(reply: str) -> dict[str, str]:
    s = UI.sub("", reply)
    marks = list(MARK.finditer(s))
    out = {}
    for i, m in enumerate(marks):
        path = m.group(1).strip().strip("`")
        # The FILE: line sits outside the code block, so markdown treats __init__ as
        # bold and hands back "init.py". Inside a block the dunder survives; only the
        # path needs putting back together.
        path = re.sub(r"(^|/)init\.py$", r"\1__init__.py", path)
        path = re.sub(r"(^|/)main\.py$", r"\1__main__.py", path) if path.endswith("/main.py") and False else path
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        body = s[m.end():end]
        head = (m.group(2) or "").strip()
        lines = body.split("\n")
        while lines and (not lines[0].strip() or LABEL.match(lines[0].strip())
                         or lines[0].strip().startswith("```")):
            lines.pop(0)
        if head and not LABEL.match(head) and not head.startswith("```"):
            lines.insert(0, head)
        while lines and (not lines[0].strip() or LABEL.match(lines[0].strip())
                         or lines[0].strip().startswith("```")):
            lines.pop(0)
        while lines and (not lines[-1].strip() or lines[-1].strip().startswith("```")):
            lines.pop()
        if ".." in path or path.startswith("/") or not lines:
            continue
        out[path] = "\n".join(lines).rstrip() + "\n"
    return out


def read(p, limit=None):
    with open(os.path.join(ROOT, p), encoding="utf-8", errors="replace") as f:
        s = f.read()
    return s[:limit] if limit else s


def contract(spec) -> str:
    kind = spec.get("kind", "M")
    n = 3 if kind == "M" else len(spec.get("milestones") or []) or 7
    tpl = TEMPLATE_M if kind == "M" else TEMPLATE_L
    nl = chr(10)
    ideas = nl.join(f"  - {i}" for i in spec.get("ideas", []))
    miles = nl.join(f"  {i+1}. {t}" for i, t in enumerate(spec.get("milestones") or []))
    title_line = ("  title     " + spec["title"]) if spec.get("title") else ""
    ticket_line = ("  ticket    " + spec["brief"]) if spec.get("brief") else ""
    miles_block = ("MILESTONE TITLES, use these exactly:" + nl + miles) if miles else ""
    return f"""You are building one exercise unit for mlsys-lab, a bank of auto-graded
exercises in low-level ML systems. Everything is in English.

THE UNIT
  id        {spec['id']}
  area      {spec['area']}
  track     {spec.get('track','')}
  tier      {spec.get('tier','T0')}
  size      {kind} — exactly {n} milestones
  gate hint {spec.get('gate_metric','')}
{title_line}
{ticket_line}

IDEAS IT MUST COVER (from the research; do not invent a different topic)
{ideas or "  (none listed — use the track name)"}
{miles_block}

THE SHAPE — a worked example of the same shape, read it and copy the structure:

--- {tpl}/project.json
{read(tpl + "/project.json", 2200)}

--- {tpl}/harness/m1.py
{read(tpl + "/harness/m1.py", 1600)}

--- last milestone checker, the safeguard
{read(tpl + ("/harness/m3.py" if kind == "M" else "/harness/m7.py"), 2600)}

--- the regression test the reference ships, which is what makes that checker pass
{read(tpl + "/reference/tests/test_regression.py", 2200)}

RULES A MACHINE CHECKS
  * reference/ must clear every milestone; skeleton/ must clear none. Both halves
    matter: a skeleton that passes means the gate measures nothing.
  * brief.md is a ticket: it states a SYMPTOM, never the diagnosis. 150+ words.
  * A gate is never wall-clock time. Use an invariant, a ratio against the
    learner's own baseline, or a comparison with an oracle you compute in
    harness/ref.py from the same inputs. Never hard-code an expected answer.
  * Deterministic: fixed seed, integer arithmetic where possible, no network,
    no binary fixtures. Generate fixtures in harness/ref.py.
  * Python and numpy only. Grading must finish in under 20 seconds.
  * No comments in code. A short docstring is fine.
  * The LAST milestone is always a safeguard: the learner writes
    tests/test_regression.py, the checker monkeypatches something in their own code
    to a broken version, and the test must then fail. The injected fault has to
    break an INVARIANT — not merely be a different valid implementation.
  * skeleton/ mirrors reference/ file for file; every function raises
    NotImplementedError. tests/test_regression.py in the skeleton raises too.
  * harness/ref.py must define every name the milestone files use from it. A
    milestone that calls ref.something the oracle does not define fails with
    AttributeError and takes the unit with it, and that is the single most
    common way these come back broken.
  * The harness directory itself is put on sys.path, so a milestone file writes
    `import ref` — never `from harness import ref`, never `import harness.ref`.
    There is no package called `harness` and no package called `reference`;
    importing one raises ModuleNotFoundError and the whole unit clears zero
    milestones. The learner's own modules are imported by their package name
    inside check(), after workdir is on the path, exactly as the example does.
  * Before you answer, walk each milestone against the reference you just wrote
    and check the numbers it would produce. "reference clears 2 of 3" is the
    other common failure, and it is always visible from the files themselves.

OUTPUT FORMAT — every file, each as:

FILE: brief.md
```markdown
...
```

FILE: project.json
```json
...
```

FILE: skeleton/<pkg>/<mod>.py
```python
...
```

...and so on for reference/, harness/ref.py, harness/m1.py … harness/m{n}.py,
skeleton/tests/test_regression.py, reference/tests/test_regression.py.

The fences are required: without them the indentation is lost in transit.
Nothing outside the FILE:/fence pairs. Do not explain."""


def syntax_error(files: dict[str, str]) -> str | None:
    """Reject a reply that cannot be Python before it reaches the disk.

    The gateway mangles indentation occasionally. Writing the file anyway spends
    a whole verification run to learn what ast.parse knows for nothing.
    """
    for name, body in files.items():
        if not name.endswith(".py"):
            continue
        try:
            ast.parse(body)
        except SyntaxError as e:
            return f"{name}: {type(e).__name__} line {e.lineno}"
    return None


def write_unit(pdir: str, files: dict[str, str]) -> int:
    for rel, body in files.items():
        dest = os.path.join(pdir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(body)
    for pkg_root in ("reference", "skeleton"):
        for cur, _dirs, fs in os.walk(os.path.join(pdir, pkg_root)):
            if any(x.endswith(".py") for x in fs) and "__init__.py" not in fs \
               and os.path.basename(cur) not in ("tests",):
                open(os.path.join(cur, "__init__.py"), "w").close()
    return len(files)


def verify(unit_id: str) -> tuple[bool, str]:
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "verify_project.py"), unit_id],
                           capture_output=True, text=True, cwd=ROOT, timeout=420)
    except subprocess.TimeoutExpired:
        return False, "verification timed out"
    txt = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)
    line = next((l.strip() for l in txt.splitlines() if unit_id in l), txt.strip()[:200])
    return ("skeleton 0/" in line and "FAIL" not in line), line


def build(unit_id: str, turns: int) -> dict:
    spec_path = os.path.join(SPECS, unit_id + ".json")
    if not os.path.isfile(spec_path):
        return {"id": unit_id, "ok": False, "why": "no spec"}
    spec = json.load(open(spec_path, encoding="utf-8"))
    pdir = os.path.join(ROOT, "projects", unit_id)
    if os.path.isdir(pdir) and os.path.isfile(os.path.join(pdir, "project.json")):
        ok, line = verify(unit_id)
        if ok:
            return {"id": unit_id, "ok": True, "skipped": True, "why": "already built"}
        # A directory left behind by a killed run is not a starting point: it is
        # a unit that never passed. Clear it and build from the contract.
        shutil.rmtree(pdir, ignore_errors=True)

    conv = f"mlsys-build-{unit_id}"
    prompt = contract(spec)
    history = []
    restarts = 0
    nudged = False
    # Whether the contract has been sent in this conversation, which is not the
    # same question as whether files exist on disk. A restart used to leave a
    # half-built directory behind, the next run read it as "already started" and
    # sent a repair prompt into a chat that had never seen the unit — the model
    # then asked for the files, which parsed to nothing and burned every turn.
    briefed = False
    have: dict[str, str] = {}
    for turn in range(turns):
        model = LADDER[min(turn // 2, len(LADDER) - 1)]
        if not briefed:
            prompt = contract(spec)
        try:
            reply = ask(model, prompt, conv, expect_files=True)
        except Exception as e:  # noqa: BLE001
            history.append(f"{model}:{type(e).__name__}: {str(e)[:60]}")
            if not os.path.isfile(os.path.join(pdir, "project.json")):
                restarts += 1
                conv = f"mlsys-build-{unit_id}-r{restarts}"
                prompt = contract(spec)
            else:
                prompt = "The previous reply did not arrive. Send the files again."
            continue

        files = parse_files(reply)
        if not files:
            # Keep the reply that produced nothing. Whether it carries a FILE:
            # marker decides whether this is a parser defect or something the
            # model never sent, and that is not worth guessing at.
            with open(f"/tmp/nofiles_{unit_id[:40]}_{turn}.txt", "w") as fh:
                fh.write(reply)
        first = not os.path.isfile(os.path.join(pdir, "project.json"))
        # A repair turn is asked to resend only what changed, so demanding the whole
        # set back would reject exactly the reply that was requested — and wiping the
        # directory first would throw away everything it did not resend.
        need_all = first
        if first:
            # A long unit does not always fit in one reply. Whatever arrived is
            # already written work; keep it and ask for the remainder rather than
            # paying for the whole set again and truncating in the same place.
            have.update(files)
            files = dict(have)
        if not files or (need_all and "project.json" not in files):
            history.append(f"{model}:no files ({len(files)})")
            if not files:
                # The contract is already in this chat, so the first miss gets a
                # nudge rather than a second copy of 6 kB — the reply that came
                # back was empty of files, not evidence the chat is unusable.
                # Only a second miss in a row starts a clean conversation.
                if nudged:
                    restarts += 1
                    conv = f"mlsys-build-{unit_id}-r{restarts}"
                    have.clear()
                    nudged = False
                    prompt = contract(spec)
                else:
                    nudged = True
                    prompt = ("Send the unit now. Every file as a line "
                              "`FILE: <path>` followed by its contents in a "
                              "fenced block, project.json first. No commentary.")
            elif first:
                got = ", ".join(sorted(files)[:12])
                prompt = ("The reply stopped early. I already have: " + got +
                          ". Send the files that are still missing — project.json "
                          "first — in the same FILE:/fenced-block format. Do not "
                          "resend what I already have.")
            else:
                prompt = ("I received no usable files. Reply again, every file as a line "
                          "`FILE: <path>` followed by the file content in a fenced block. "
                          "Nothing else, no commentary.")
            continue

        broken = syntax_error(files)
        if broken:
            # One file failed to parse, usually because its newlines were lost
            # on the way back. Everything else in the reply is good work, so it
            # is kept and only the broken file is asked for again — resending
            # the contract discards a dozen files to repair one.
            history.append(f"{model}:{broken}")
            name = broken.split(":")[0]
            have.update({k: v for k, v in files.items() if k != name})
            prompt = ("`" + name + "` did not parse as Python (" + broken +
                      "). Send that one file again, complete, as `FILE: " + name +
                      "` followed by a fenced block. Nothing else.")
            continue

        if first:
            shutil.rmtree(pdir, ignore_errors=True)
        try:
            n = write_unit(pdir, files)
        except Exception as e:  # noqa: BLE001
            history.append(f"{model}:write {type(e).__name__}: {str(e)[:60]}")
            continue

        ok, line = verify(unit_id)
        if ok:
            return {"id": unit_id, "ok": True, "model": model, "turn": turn + 1,
                    "files": n, "history": history, "line": line}
        history.append(f"{model}:{line[-90:]}")
        prompt = (f"`python3 tools/verify_project.py {unit_id}` reports:\n\n    {line}\n\n"
                  "Fix it and resend ONLY the files that change, in the same "
                  "FILE:/fenced-block format. Remember: the reference has to clear every "
                  "milestone and the skeleton has to clear none, so if the skeleton is "
                  "passing, the gate is not measuring anything and needs to be stricter.")

    shutil.rmtree(pdir, ignore_errors=True)
    return {"id": unit_id, "ok": False, "why": "; ".join(history[-3:])}


def queue(a) -> list[str]:
    out = []
    for f in sorted(os.listdir(SPECS)):
        if not f.endswith(".json"):
            continue
        spec = json.load(open(os.path.join(SPECS, f), encoding="utf-8"))
        if os.path.isfile(os.path.join(ROOT, "projects", spec["id"], "project.json")):
            continue
        if a.tier and not spec.get("tier", "").startswith(a.tier):
            continue
        if a.area and a.area not in spec.get("area", ""):
            continue
        if a.kind and spec.get("kind") != a.kind.upper():
            continue
        out.append(spec["id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--tier", default=None)
    ap.add_argument("--area", default=None)
    ap.add_argument("--kind", default=None, choices=["m", "l"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--turns", type=int, default=4)
    a = ap.parse_args()

    ids = a.ids or queue(a)
    if a.limit:
        ids = ids[:a.limit]
    if not ids:
        print("nothing to build")
        return 0
    if not a.ids and not a.limit and not a.all:
        print(f"{len(ids)} units queued; pass --all to build them")
        return 0

    say(f"{len(ids)} units · {min(a.j,10)} at a time · up to {a.turns} turns each")
    ok = bad = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=min(a.j, 10)) as ex:
        futs = {ex.submit(build, i, a.turns): i for i in ids}
        for fut in cf.as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                import traceback
                r = {"id": futs[fut], "ok": False,
                     "why": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc()[-400:]}
            log(dict(r, at=time.strftime("%H:%M:%S")))
            if r["ok"] and not r.get("skipped"):
                ok += 1
                say(f"  ok   {r['id']:<50} {r.get('model','-')} turn {r.get('turn','-')} "
                    f"{r.get('files','')} files")
            elif not r["ok"]:
                bad += 1
                say(f"  FAIL {r['id']:<50} {str(r.get('why',''))[:90]}")
    say(f"\n{ok} built, {bad} failed, {(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
