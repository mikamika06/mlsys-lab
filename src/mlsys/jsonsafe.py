"""JSON that both Python and JavaScript can read.

`json.dumps` writes bare `Infinity` and `NaN` for non-finite floats. Python reads
them back happily; `JSON.parse` in the browser and in the VS Code extension does
not, and rejects the whole document. Since a failing metric is very often `inf` —
that is what a grader returns when the submission raises — the reader would see a
parse error instead of a FAIL verdict exactly when the learner needs the verdict.

So non-finite values are emitted as the strings "Infinity", "-Infinity" and
"NaN". They stay valid JSON, they keep their meaning, and the UI formats them.
"""
from __future__ import annotations

import json
import math


def clean(o):
    """Replace every non-finite float with a string, recursively."""
    if isinstance(o, float):
        if math.isnan(o):
            return "NaN"
        if math.isinf(o):
            return "Infinity" if o > 0 else "-Infinity"
        return o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def dumps(o, **kw):
    """json.dumps, but the output always parses in a browser."""
    kw.setdefault("allow_nan", False)     # anything we missed becomes a loud error
    return json.dumps(clean(o), **kw)
