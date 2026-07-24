import importlib
import importlib.abc
import importlib.util
import sys
import types
import uuid


def _oracle(name, source, attr):
    class OracleLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return types.ModuleType(spec.name)

        def exec_module(self, module):
            exec(source, module.__dict__)

    class OracleFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == name:
                return importlib.util.spec_from_loader(fullname, OracleLoader())
            return None

    finder = OracleFinder()
    sys.meta_path.insert(0, finder)
    sys.modules.pop(name, None)
    try:
        module = importlib.import_module(name)
        return getattr(module, attr)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.pop(name, None)


def grade(sol, fx) -> dict:
    cases = [
        (
            "virtual_" + uuid.uuid4().hex,
            "value = 10 + 5",
            "value",
        ),
        (
            "virtual_" + uuid.uuid4().hex,
            "value = __name__ + ':' + __spec__.name",
            "value",
        ),
        (
            "virtual_" + uuid.uuid4().hex,
            "def f():\n    return (__name__, __spec__.name)\n",
            "f",
        ),
        (
            "virtual_" + uuid.uuid4().hex,
            "value = (__spec__.loader is not None)",
            "value",
        ),
    ]

    ok = 1.0
    for name, source, attr in cases:
        try:
            expected = _oracle(name, source, attr)
            got = sol.materialize_attr(name, source, attr)
        except Exception:
            ok = 0.0
            break

        if callable(expected):
            if not callable(got) or got() != expected():
                ok = 0.0
                break
        elif got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
