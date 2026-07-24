"""Reference solution for `pyt-circular-import-failure-predictor`.

A small, faithful simulator of CPython's import machinery, restricted to
three statement kinds (see task.md): `bind`, `import_module`, and
`from_import`. It mirrors exactly what `sys.modules` + module namespaces
do during a real import:

  * A module is only ever executed once. If it's already in `sys_modules`
    (even mid-execution, i.e. "partially initialized"), re-importing it
    just returns the SAME (possibly incomplete) namespace -- no re-run,
    no error.
  * `import other` never inspects `other`'s namespace, so it can never
    raise by itself: it just binds a reference to whatever `other`'s
    module object currently is (complete or partial).
  * `from other import name` DOES inspect `other`'s namespace at that
    exact moment. If `name` isn't bound yet (because `other` is still
    mid-execution, stuck earlier in its own body), that's exactly
    Python's real `ImportError: cannot import name ... from partially
    initialized module ...`.
"""
from __future__ import annotations


def predict_import_result(modules: dict, entry: str) -> bool:
    sys_modules: dict = {}

    def run(name: str) -> dict:
        if name in sys_modules:
            return sys_modules[name]          # partial or complete -- reuse as-is

        namespace: set = set()
        sys_modules[name] = {"namespace": namespace}

        for op in modules[name]:
            kind = op[0]
            if kind == "bind":
                namespace.add(op[1])
            elif kind == "import_module":
                other = op[1]
                if other not in sys_modules:
                    run(other)
                namespace.add(other)
            elif kind == "from_import":
                other, wanted = op[1], op[2]
                if other not in sys_modules:
                    run(other)
                other_ns = sys_modules[other]["namespace"]
                if wanted not in other_ns:
                    raise ImportError(
                        f"cannot import name {wanted!r} from {other!r}"
                    )
                namespace.add(wanted)
            else:
                raise ValueError(f"unknown op kind: {kind!r}")

        return sys_modules[name]

    try:
        run(entry)
        return True
    except ImportError:
        return False
