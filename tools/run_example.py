#!/usr/bin/env python3
"""Run one statement's example against its reference and print what it prints.

Kept as its own process so a rewritten example that loops for ever costs a
timeout rather than the pipeline.

    python3 tools/run_example.py <reference.py> <example.py>
"""
import io
import contextlib
import runpy
import sys


def main():
    ref, ex = sys.argv[1], sys.argv[2]
    ns = runpy.run_path(ref)
    ns = {k: v for k, v in ns.items() if not k.startswith("__")}
    with open(ex, encoding="utf-8") as f:
        code = f.read()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<example>", "exec"), ns)
    sys.stderr.write("OK\n")
    print(buf.getvalue(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
