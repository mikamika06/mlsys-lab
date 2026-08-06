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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_corrupted_config": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import amx.config as cfg_mod
    orig_encode = cfg_mod.encode_tilecfg

    def faulty_encode(tile_specs, palette_id=1, start_row=0):
        buf = bytearray(64)
        buf[0] = palette_id & 0xFF
        buf[1] = start_row & 0xFF
        for t_id, spec in tile_specs.items():
            if 0 <= t_id <= 7:
                colsb = spec.get("bytes_per_row", 0)
                rows = spec.get("rows", 0)
                buf[16 + t_id * 2] = rows & 0xFF
                buf[16 + t_id * 2 + 1] = (rows >> 8) & 0xFF
                buf[48 + t_id] = colsb & 0xFF
        return bytes(buf)

    cfg_mod.encode_tilecfg = faulty_encode
    import amx
    amx.config.encode_tilecfg = faulty_encode

    try:
        out["catches_corrupted_config"] = 0.0 if _survives(path) else 1.0
    finally:
        cfg_mod.encode_tilecfg = orig_encode
        amx.config.encode_tilecfg = orig_encode

    return out
