#!/usr/bin/env python3
"""Turn a grader's JSON on stdin into one line naming each gate and its value.

"reference does not pass gates" is a failure nobody can act on — least of all from a
CI log produced on a different architecture from the one you are sitting at. This
prints what the numbers actually were.

    ... --json | python3 tools/gate_detail.py
"""
import json
import sys


def main() -> int:
    try:
        d = json.load(sys.stdin)
    except Exception:
        print("no JSON from the grader")
        return 0

    if d.get("error"):
        last = (d["error"].strip().splitlines() or ["?"])[-1]
        print("raised: " + last[:160])
        return 0

    parts = []
    for g in d.get("gates", []):
        v = g.get("value")
        shown = f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
        mark = "" if g.get("ok") else "  <-- FAILED"
        parts.append(f"{g['metric']}={shown} (gate {g['op']}{g['threshold']:g}){mark}")
    print(" | ".join(parts) or "no gates reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
