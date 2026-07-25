#!/usr/bin/env python3
"""Repair LaTeX in task statements that KaTeX cannot render.

A statement whose formula fails to parse shows the reader raw source, which is
worse than no formula at all. The failures in this bank fall into a handful of
mechanical classes, so they are rewritten here rather than one file at a time:

  \\text{max_abs_err}   an underscore is still math-active inside \\text{}, so a
                       metric or identifier name has to be \\texttt{} with the
                       underscores escaped
  x.__dict__           the same thing, seen as a double subscript
  \\beta^\\*             `*` needs no backslash in math mode
  \\text{# ...}         `#` is a macro parameter character and must be escaped
  \\ref                 no bibliography here; it was meant as a label
  psmallmatrix         not in KaTeX; pmatrix is the equivalent that exists
  a stray `$`          a display block that swallowed the closing delimiter

Run with --apply to write. Without it, it only reports.
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MATH = re.compile(r"(\$\$)([\s\S]*?)(\$\$)|(\$)([^$\n]+)(\$)")

# A dunder is unambiguous: nothing in real maths is spelled __name__.
DUNDER = re.compile(r"(?<![\w\\])__\w+__(?![\w])")


def esc_underscores(body):
    """Escape only the underscores that are not escaped already.

    Doing it twice yields `\\\\_`, which KaTeX reads as a line break followed by an
    underscore — just as broken as the original.
    """
    return re.sub(r"(?<!\\)_", r"\\_", body)


def fix_span(src):
    """Rewrite one math span, conservatively.

    Only transformations that cannot change what the formula MEANS are applied.
    An earlier version of this script tried to recognise identifiers by looking
    for an underscore between word characters, and destroyed `\max_{i,j}`,
    `\rVert_2` and `L_{b,t,v}` — all of which are ordinary subscripts. A regex
    cannot tell a subscript from an identifier, so it no longer tries: anything
    ambiguous is left for a person.
    """
    out = src

    # \text{max_abs_err} — inside \text an underscore is still math-active, so it
    # is always a mistake there; \texttt is the right box for a name.
    def textish(m):
        body = m.group(2)
        if "_" not in body and "#" not in body:
            return m.group(0)
        return r"\texttt{" + esc_underscores(body).replace("#", r"\#") + "}"
    out = re.sub(r"\\(text|mathrm|mathit)\{([^{}]*)\}", textish, out)

    # \texttt{keep_ratio} — right box, unescaped content
    out = re.sub(r"\\texttt\{([^{}]*)\}",
                 lambda m: r"\texttt{" + esc_underscores(m.group(1)) + "}", out)

    # bare dunders: x.__dict__, "__set__", .__name__
    out = DUNDER.sub(lambda m: r"\texttt{" + esc_underscores(m.group(0)) + "}", out)

    # `*` is not a control sequence
    out = re.sub(r"\^\{?\\\*\}?", "^{*}", out)
    out = re.sub(r"_\{?\\\*\}?", "_{*}", out)

    out = re.sub(r"\\ref\b(?:\{[^}]*\})?", r"\\mathrm{ref}", out)
    out = out.replace("psmallmatrix", "pmatrix")
    return out


def katex_ok(spans):
    """Ask the real KaTeX which spans parse. spans: [(src, display)]."""
    import json
    js = ROOT / "editor" / "media" / "katex" / "katex.min.js"
    prog = (
        "const k=require(%s);const inp=JSON.parse(process.argv[1]);"
        "console.log(JSON.stringify(inp.map(([s,d])=>{"
        "try{k.renderToString(s,{displayMode:d,throwOnError:true});return null;}"
        "catch(e){return e.message.slice(0,120);}})));" % json.dumps(str(js))
    )
    r = subprocess.run(["node", "-e", prog, json.dumps(spans)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("katex probe failed: " + r.stderr[:400])
    return json.loads(r.stdout)


def spans_of(text):
    out = []
    for m in MATH.finditer(text):
        display = m.group(1) is not None
        out.append((m.group(2) if display else m.group(5), display, m.span()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    files = sorted((ROOT / "tasks").glob("*/task.md"))
    todo = []
    for f in files:
        t = f.read_text(encoding="utf-8")
        sp = spans_of(t)
        if not sp:
            continue
        errs = katex_ok([[s, d] for s, d, _ in sp])
        for (src, disp, span), e in zip(sp, errs):
            if e:
                todo.append((f, span, src, disp, e))

    print(f"broken math spans: {len(todo)} in {len({f for f, *_ in todo})} tasks")
    if not todo:
        return

    fixed = unfixed = 0
    per_file = {}
    for f, span, src, disp, e in todo:
        new = fix_span(src)
        ok = katex_ok([[new, disp]])[0] is None
        if ok:
            fixed += 1
            per_file.setdefault(f, []).append((span, src, new))
        else:
            unfixed += 1
            print(f"  STILL BROKEN {f.parent.name}\n     {e[:70]}\n     {src.strip()[:100]}")

    print(f"\nrepairable: {fixed} | needs a human: {unfixed}")
    if not a.apply:
        print("dry run — re-run with --apply")
        return

    for f, edits in per_file.items():
        t = f.read_text(encoding="utf-8")
        for (s, e), old, new in sorted(edits, key=lambda x: -x[0][0]):
            body = t[s:e]
            t = t[:s] + body.replace(old, new, 1) + t[e:]
        f.write_text(t, encoding="utf-8")
    print(f"rewrote {len(per_file)} files")


if __name__ == "__main__":
    main()
