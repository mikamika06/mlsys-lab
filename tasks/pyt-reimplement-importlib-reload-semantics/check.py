import importlib
import os
import sys
import tempfile
import types
import uuid


def _oracle(source_before, source_after):
    with tempfile.TemporaryDirectory() as d:
        name = "arena_reload_" + uuid.uuid4().hex
        path = os.path.join(d, name + ".py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source_before)
        sys.path.insert(0, d)
        try:
            importlib.invalidate_caches()
            mod = importlib.import_module(name)
            before_id = id(mod)

            with open(path, "w", encoding="utf-8") as f:
                f.write(source_after)

            importlib.invalidate_caches()
            importlib.reload(mod)

            return (id(mod) == before_id, mod.value)
        finally:
            sys.path.remove(d)
            sys.modules.pop(name, None)


def grade(sol, fx) -> dict:
    cases = [
        (
            "value = 1\n",
            "value = 10\n",
        ),
        (
            "value = 'old'\nextra = 3\n",
            "value = 'new'\n",
        ),
        (
            "value = 7\n",
            "value = 7 * 6\n",
        ),
    ]

    ok = 1.0
    for before, after in cases:
        try:
            module = types.ModuleType("candidate")
            module.__dict__["value"] = None
            before_id = id(module)
            got = sol.reload_module_semantics(module, after)
        except Exception:
            ok = 0.0
            break

        expected = _oracle(before, after)
        candidate_identity = id(module) == before_id
        candidate_result = (candidate_identity, getattr(module, "value", None))

        if got != expected or candidate_result != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
