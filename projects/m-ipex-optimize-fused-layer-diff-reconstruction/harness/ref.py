class MockSubModule:

    def __init__(self, class_name):
        self.__name = class_name
        self.__class__ = type(class_name, (object,), {})


class DummyModel:

    def __init__(self, modules):
        self.modules_map = modules

    def named_modules(self):
        return self.modules_map.items()


def generate_sample_models():
    orig_map = {
        "": MockSubModule("Sequential"),
        "fc1": MockSubModule("Linear"),
        "relu1": MockSubModule("ReLU"),
        "conv1": MockSubModule("Conv2d"),
        "bn1": MockSubModule("BatchNorm2d"),
        "fc2": MockSubModule("Linear"),
    }
    opt_map = {
        "": MockSubModule("Sequential"),
        "fc1": MockSubModule("IPEXLinearFused"),
        "relu1": MockSubModule("ReLU"),
        "conv1": MockSubModule("IPEXConv2dFused"),
        "bn1": MockSubModule("BatchNorm2d"),
        "fc2": MockSubModule("IPEXLinearFused"),
    }
    return DummyModel(orig_map), DummyModel(opt_map)


def mock_model_runner(x, mode="autocast"):
    val = sum(x) if isinstance(x, (list, tuple)) else x
    if mode == "autocast":
        # Simulate slight overhead for AMP conversion
        res = 0
        for _ in range(100):
            res += val
        return res
    else:
        # Simulate optimized IPEX execution path
        res = 0
        for _ in range(20):
            res += val
        return res
