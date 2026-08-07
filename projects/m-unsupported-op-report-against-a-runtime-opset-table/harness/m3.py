import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_topo": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct plan: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import opset.drift as d
    good_migrate = d.migrate_squeeze_11

    def bad_migrate(nodes):
        out_nodes = []
        for n in nodes:
            if n["op_type"] == "Squeeze" and "axes" in n.get("attributes", {}):
                axes = n["attributes"]["axes"]
                new_attrs = {k: v for k, v in n["attributes"].items() if k != "axes"}
                const_name = n["name"] + "_axes"
                new_n = dict(n)
                new_n["attributes"] = new_attrs
                new_n["inputs"] = list(n["inputs"]) + [const_name + "_out"]
                out_nodes.append(new_n)
                out_nodes.append({
                    "name": const_name,
                    "op_type": "Constant",
                    "inputs": [],
                    "outputs": [const_name + "_out"],
                    "attributes": {"value": axes}
                })
            else:
                out_nodes.append(dict(n))
        return out_nodes

    d.migrate_squeeze_11 = bad_migrate
    try:
        out["catches_bad_topo"] = 0.0 if _survives(path) else 1.0
    finally:
        d.migrate_squeeze_11 = good_migrate

    return out
