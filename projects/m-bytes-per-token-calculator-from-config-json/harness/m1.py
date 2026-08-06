import importlib.util
import os
import ref
from kvcalc.calc import bytes_per_token


def _load_ref_calc():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "reference", "kvcalc", "calc.py")
    spec = importlib.util.spec_from_file_location("ref_calc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bytes_per_token


def check(workdir):
    out = {"bytes_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    ref_bytes = _load_ref_calc()
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref_bytes(cfg, 2)
        try:
            got = bytes_per_token(cfg, 2)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got}, reference {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} raised {type(e).__name__}"
    out["bytes_matched"] = float(ok)
    return out
