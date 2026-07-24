import importlib
import sys


def cached_import_count(module_name, module_dir):
    old_path = list(sys.path)
    try:
        sys.modules.pop(module_name, None)
        sys.path.insert(0, module_dir)
        module = importlib.import_module(module_name)
        importlib.import_module(module_name)
        return int(module.EXECUTIONS)
    finally:
        sys.path[:] = old_path
        sys.modules.pop(module_name, None)
