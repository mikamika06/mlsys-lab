import importlib
import importlib.abc
import importlib.util
import sys
import types


def materialize_attr(name, source, attr):
    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return types.ModuleType(spec.name)

        def exec_module(self, module):
            exec(source, module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == name:
                return importlib.util.spec_from_loader(fullname, Loader())
            return None

    finder = Finder()
    sys.meta_path.insert(0, finder)
    sys.modules.pop(name, None)
    try:
        module = importlib.import_module(name)
        return getattr(module, attr)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.pop(name, None)
