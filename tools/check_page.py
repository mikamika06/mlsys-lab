#!/usr/bin/env python3
"""Refuse to publish a concept page that is thin, or that lies.

The bar comes from measuring what actually holds page one for these terms: median
1,081 words, some code, no interactive element anywhere. Beating that does not need
length — it needs the one thing no competing page has, which is a number produced
by this repo that the reader can regenerate. So a page without a measured table and
the command that reproduces it does not ship, and the check is mechanical rather
than a good intention.

It also resolves every relative link, because a published page with a broken link
to a task that does not exist is worse than no page.

    python3 tools/check_page.py concepts/*.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MIN_WORDS, MAX_WORDS = 600, 1400          # prose only; tables and code excluded
MIN_INTERNAL, MAX_INTERNAL = 8, 25
MIN_REFS = 2

FENCE = re.compile(r"```.*?```", re.S)
TABLE_ROW = re.compile(r"^\|.*\|.*\|", re.M)
MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def check(p: Path) -> list[str]:
    t = p.read_text(encoding="utf-8")
    bad: list[str] = []

    fm = FRONT.match(t)
    if not fm:
        bad.append("no YAML front matter")
    else:
        for key in ("title", "description", "datePublished", "dateModified"):
            if not re.search(rf"^{key}:", fm.group(1), re.M):
                bad.append(f"front matter missing `{key}`")

    prose = FENCE.sub("", t)
    prose = TABLE_ROW.sub("", prose)
    words = len(prose.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        bad.append(f"{words} prose words, want {MIN_WORDS}-{MAX_WORDS}")

    if not TABLE_ROW.search(t):
        bad.append("no measured table — the whole point of the page")
    if "Reproduce it" not in t:
        bad.append("no `Reproduce it` section: a number nobody can regenerate is a claim")
    if not re.search(r"```(bash|python|cpp|cuda|c\+\+)", t):
        bad.append("no runnable code block")
    if "mlsys grade" not in t:
        bad.append("no `mlsys grade` line: the page must lead to an exercise")

    # the exercise must come after the first measured table, not before it
    first_table = TABLE_ROW.search(t)
    grade_at = t.find("mlsys grade")
    if first_table and grade_at != -1 and grade_at < first_table.start():
        bad.append("`mlsys grade` appears before the first table — reads as an advert")

    # A reference is a numbered item in the References section that cites a URL.
    # The URL is usually on a continuation line, so count per item, not per line.
    tail = t.split("## References", 1)
    if len(tail) < 2:
        bad.append("no `## References` section")
    else:
        items = re.split(r"^\d+\. ", tail[1], flags=re.M)[1:]
        cited = [i for i in items if "http" in i]
        if len(cited) < MIN_REFS:
            bad.append(f"{len(cited)} references with a URL, want >= {MIN_REFS}")

    internal = external = 0
    for text, href in MD_LINK.findall(t):
        if href.startswith("http"):
            external += 1
            continue
        if href.startswith("#"):
            continue
        internal += 1
        target = (p.parent / href.split("#")[0]).resolve()
        if not target.exists():
            bad.append(f"broken link: [{text}]({href})")
    if not MIN_INTERNAL <= internal <= MAX_INTERNAL:
        bad.append(f"{internal} internal links, want {MIN_INTERNAL}-{MAX_INTERNAL}")

    return bad


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        files = sorted((ROOT / "concepts").glob("*.md"))
        files = [f for f in files if f.name != "README.md"]
    if not files:
        print("no pages to check")
        return 0

    failed = 0
    for f in files:
        bad = check(f)
        if bad:
            failed += 1
            print(f"FAIL {f}")
            for b in bad:
                print(f"       {b}")
        else:
            print(f"ok   {f.name}")
    print(f"\n{len(files) - failed}/{len(files)} pages pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
