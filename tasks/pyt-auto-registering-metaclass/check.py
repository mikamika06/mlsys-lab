def build_registry():
    class RegistryMeta(type):
        registry = {}

        def __new__(mcls, name, bases, namespace):
            cls = super().__new__(mcls, name, bases, namespace)
            if name != "BasePlugin":
                mcls.registry[name] = cls.__qualname__
            return cls

    class BasePlugin(metaclass=RegistryMeta):
        pass

    class JsonPlugin(BasePlugin):
        pass

    class CsvPlugin(BasePlugin):
        pass

    class XmlPlugin(BasePlugin):
        pass

    return dict(RegistryMeta.registry)


def _oracle_registry():
    return build_registry()


def grade(sol, fx) -> dict:
    try:
        got = sol.build_registry()
    except Exception:
        return {"exact_match": 0.0}

    expected = _oracle_registry()
    return {"exact_match": 1.0 if got == expected else 0.0}
