#!/usr/bin/env python3
"""Generate RESOURCES.md — a flat menu of everywhere else you can practise this material.

LANDSCAPE.md already surveys the same 141 resources, but it is organised as "our 14 areas,
and how each compares to us". That answers a question about this project. It does not answer
"I want to practise quantization, where do I go", because getting there means reading past
paragraphs about how we differ.

So this is the other cut of the same data: grouped by what a resource IS and whether it checks
your work, with the cost, whether it is still alive, and one sentence on what it is. No
comparison to this bank anywhere. Deduplicated by URL, because a resource that serves several
areas appeared once per area in the survey.

    python3 tools/gen_resources.py <survey.json> <urlcheck.json>
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Ordered: the ones that tell you whether you got it right come first, because that is the
# thing that is scarce. Everything after "read only" is material, not practice.
GRADED_ORDER = [
    ("auto-graded", "Graded automatically",
     "You submit, a machine gives you a verdict. The scarcest category by far."),
    ("self-checked-tests", "Ships tests you run yourself",
     "No submission, but the tests are there and they are real — you find out."),
    ("reference-only", "Reference code to read",
     "Implementations to study. Nothing checks you."),
    ("no", "Reading, tools and reference",
     "Books, papers, docs and interactive explainers. Nothing checks you."),
]

AREA_TITLE = {
    "python-core": "Deep Python", "cpp-core": "Deep C++", "cpu-perf": "CPU performance",
    "gpu-cuda": "GPU / CUDA", "numeric-tensors": "Numerics and tensors",
    "algorithms-scratch": "Algorithms from scratch", "llm-internals": "LLM internals",
    "llm-systems": "LLM systems", "rw-applied-quantization": "Applied quantization",
    "rw-attention-and-kv": "Attention and KV cache",
    "rw-compilation-and-export": "Compilation and export",
    "rw-batching-and-serving": "Batching and serving",
    "rw-memory-and-offload": "Memory and offload",
    "rw-sparsity-pruning-distillation": "Sparsity, pruning, distillation",
}

DEAD = re.compile(r"archiv|dead|dormant|stalled|unmaintained|no commits|frozen|read-only", re.I)

# The survey's descriptions occasionally record how the surveyor checked something —
# "I read tests/adapters.py directly: it requires…". That is methodology, useful in the
# survey and out of place in a catalogue someone reads to pick a resource.
FIRST_PERSON = [
    # "I read tests/adapters.py directly: it requires X" -> "tests/adapters.py requires X"
    (re.compile(r"I read (\S+) directly: it (\w+)"), r"\1 \2"),
    # "; I fetched A and B directly and confirmed both are live with C." -> " with C."
    (re.compile(r"[;,] I fetched .*? and confirmed (?:both|it) (?:are|is) live with "), " with "),
    # anything left: drop the clause rather than leave a first-person report in a catalogue
    (re.compile(r"(?:^|(?<=[.;] ))I (?:read|fetched|checked|confirmed|verified)\b[^.;]*[.;]\s*"), ""),
]


def clean_description(text: str) -> str:
    for pat, repl in FIRST_PERSON:
        text = pat.sub(repl, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def area_of(raw: str) -> str:
    for k, v in AREA_TITLE.items():
        if k in raw:
            return v
    return raw


def main() -> int:
    survey = json.loads(pathlib.Path(sys.argv[1]).read_text())
    checks = json.loads(pathlib.Path(sys.argv[2]).read_text())

    # one entry per URL, remembering every area it serves
    byurl: dict[str, dict] = {}
    for a in survey:
        for r in a["resources"]:
            e = byurl.setdefault(r["url"], dict(r, areas=[]))
            e["areas"].append(area_of(a["area"]))

    out: list[str] = []
    W = out.append

    W("# Where else to practise this")
    W("")
    W("Everywhere else worth your time on the same material, in one list. Grouped by whether")
    W("it checks your work, because that is the part that is scarce — of the resources below,")
    n_auto = sum(1 for e in byurl.values() if e["graded"] == "auto-graded")
    W(f"**{n_auto} give you an automatic verdict** and the rest do not.")
    W("")
    W("Every link was fetched when this was written and then HTTP-checked separately. Dates are")
    W("the most recent activity that could be verified; where a project is archived or dormant it")
    W("says so, because a dead project is still worth reading and worth knowing is dead.")
    W("")
    W("This is the practical cut. [`LANDSCAPE.md`](LANDSCAPE.md) covers the same ground organised")
    W("by this bank's own areas, with a verdict on where this bank does and does not add anything.")
    W("")
    W("Last checked **2026-07-26**.")
    W("")

    # ---- summary ----
    W("| | count |")
    W("|---|---:|")
    for key, title, _ in GRADED_ORDER:
        W(f"| {title} | {sum(1 for e in byurl.values() if e['graded'] == key)} |")
    W(f"| **Total** | **{len(byurl)}** |")
    W("")

    free = sum(1 for e in byurl.values() if e["cost"] == "free")
    W(f"{free} of {len(byurl)} are free. Paid and freemium entries say so on the line.")
    W("")

    for key, title, blurb in GRADED_ORDER:
        group = [e for e in byurl.values() if e["graded"] == key]
        if not group:
            continue
        W(f"## {title}")
        W("")
        W(blurb)
        W("")
        for e in sorted(group, key=lambda x: x["name"].lower()):
            name = e["name"]
            bits = [e["cost"]]
            if e["kind"] not in ("graded-platform",):
                bits.append(e["kind"].replace("-", " "))
            dead = " · **dormant or archived**" if DEAD.search(e["last_activity"]) else ""
            eff = checks.get(e["url"], "")
            flag = ""
            if eff and not eff.startswith(("200", "30")):
                code = eff.split()[0]
                flag = (" · *host blocks automated fetchers; the resource is live*"
                        if code == "403" else " · *did not respond when last checked*")
            W(f"### [{name}]({e['url']})")
            W("`" + "` · `".join(bits) + f"`  ")
            W(f"{e['size']} · last activity {e['last_activity']}{dead}{flag}  ")
            W(", ".join(sorted(set(e["areas"]))))
            W("")
            W(clean_description(e["description"]))
            W("")
        W("")

    W("## How this was built")
    W("")
    W("One research pass per area, each required to fetch every URL before reporting it and")
    W("explicitly forbidden from padding a short list with tangential material — an area with")
    W("three real resources returns three. Every URL was then checked independently of the agent")
    W("that found it.")
    W("")
    W("Two limits worth knowing. It is a snapshot: several of these are dormant and any of them")
    W("can change. And absence of evidence is weak evidence — something that exists but is not")
    W("findable through GitHub search, Google, or the bibliographies of the standard books in an")
    W("area would not have been found. A missing or mischaracterised entry is a bug; open an issue.")
    W("")
    W("Regenerate with `python3 tools/gen_resources.py` after re-surveying.")

    p = ROOT / "RESOURCES.md"
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {p} — {len(byurl)} unique resources, {n_auto} auto-graded, "
          f"{len(' '.join(out).split())} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
