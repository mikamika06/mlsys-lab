#!/usr/bin/env python3
"""Turn the Part-2 plan into one spec file per unit, so a fleet can claim them.

The plan is prose; a builder needs a contract. This reads PART2-PLAN.md and emits
`tools/specs2/<id>.json`, each carrying what the builder is not allowed to invent:
the area, the track, the tier, the gate metric the research proposed, and the
research ideas the unit is made of.

    python3 tools/gen_part2_specs.py [path-to-plan] [--limit N] [--area rw3-...]
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.expanduser("~/compression-arena/docs/research2/PART2-PLAN.md")
OUT = os.path.join(ROOT, "tools", "specs2")

AREA = re.compile(r"^# (rw[23]-[a-z0-9-]+)\s*$", re.M)
MSEC = re.compile(r"^## M\b.*\((\d+)\)\s*$", re.M)
LSEC = re.compile(r"^## L\b.*\((\d+)\)\s*$", re.M)
MROW = re.compile(r"^- `(m-[a-z0-9-]+)` · (T\d(?:\+T\d)?) · `([^`]+)` — (.+)$", re.M)
LHEAD = re.compile(r"^### \d+\. `(p-[a-z0-9-]+)` · (T\d(?:\+T\d)?)\s*$", re.M)
TRACK = re.compile(r"^\*\*(.+?)\*\* — \d+\s*$", re.M)
CONT = "; \u0434\u0430\u043b\u0456: "


def sections(text):
    """(area, body) for each area heading."""
    hits = list(AREA.finditer(text))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        yield m.group(1), text[m.end():end]


def parse_m(body):
    """M rows, each tagged with the track heading it sat under."""
    ms = MSEC.search(body)
    if not ms:
        return []
    tail = body[ms.end():]
    marks = [(m.start(), m.group(1)) for m in TRACK.finditer(tail)]
    out = []
    for row in MROW.finditer(tail):
        track = ""
        for pos, name in marks:
            if pos < row.start():
                track = name
            else:
                break
        # The plan document is a private file written in Ukrainian; this literal is
        # its list separator, not user-facing text, and changing it here would only
        # stop the parser from finding anything.
        head, _, rest = row.group(4).partition(CONT)
        ideas = [head] + ([x.strip() for x in rest.split("; ") if x.strip()] if rest else [])
        out.append({"id": row.group(1), "tier": row.group(2), "gate_metric": row.group(3),
                    "track": track, "ideas": ideas})
    return out


def parse_l(body):
    ls = LSEC.search(body)
    if not ls:
        return []
    end = MSEC.search(body)
    tail = body[ls.end():end.start() if end else len(body)]
    heads = list(LHEAD.finditer(tail))
    out = []
    for i, h in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(tail)
        block = tail[h.end():stop]
        title = ""
        tm = re.search(r"^\*\*(.+?)\*\*\s*$", block, re.M)
        if tm:
            title = tm.group(1)
        brief = ""
        bm = re.search(r"^> (.+)$", block, re.M)
        if bm:
            brief = bm.group(1)
        miles = re.findall(r"^\d+\. (.+)$", block, re.M)
        out.append({"id": h.group(1), "tier": h.group(2), "title": title,
                    "brief": brief, "milestones": miles})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", nargs="?", default=PLAN)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--area", default=None)
    ap.add_argument("--kind", choices=["m", "l", "both"], default="both")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.plan):
        print(f"no plan at {a.plan}", file=sys.stderr)
        return 2
    with open(a.plan, encoding="utf-8") as f:
        text = f.read()

    os.makedirs(OUT, exist_ok=True)
    written = 0
    counts = {"m": 0, "l": 0}
    for area, body in sections(text):
        if a.area and area != a.area:
            continue
        units = []
        if a.kind in ("m", "both"):
            units += [dict(u, kind="M", area=area) for u in parse_m(body)]
        if a.kind in ("l", "both"):
            units += [dict(u, kind="L", area=area) for u in parse_l(body)]
        for u in units:
            if a.limit and written >= a.limit:
                break
            u["milestone_count"] = 3 if u["kind"] == "M" else 7
            path = os.path.join(OUT, u["id"] + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(u, f, ensure_ascii=False, indent=1)
                f.write("\n")
            counts[u["kind"].lower()] += 1
            written += 1
    print(f"wrote {written} specs to tools/specs2  (M {counts['m']}, L {counts['l']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
