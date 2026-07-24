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
