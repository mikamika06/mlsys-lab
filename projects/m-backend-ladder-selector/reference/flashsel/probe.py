import importlib
import sys

def probe_backend(name):
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, "is_available") and callable(mod.is_available):
            return bool(mod.is_available())
        return True
    except Exception:
        return False
