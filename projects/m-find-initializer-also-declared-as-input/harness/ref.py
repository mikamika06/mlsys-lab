class DummyEntry:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class DummyTensor:
    def __init__(self, name, raw_data=None, dims=None, external_data=None):
        self.name = name
        self.raw_data = raw_data or b""
        self.dims = dims or []
        self.external_data = external_data or []


class DummyValueInfo:
    def __init__(self, name):
        self.name = name


class DummyGraph:
    def __init__(self, initializer, input_list):
        self.initializer = initializer
        self.input = input_list


class DummyModel:
    def __init__(self, graph):
        self.graph = graph


def generate_test_case_1():
    init = [DummyTensor("w1"), DummyTensor("w2"), DummyTensor("bias")]
    inp = [DummyValueInfo("w1"), DummyValueInfo("input_tensor")]
    return DummyModel(DummyGraph(init, inp))


def generate_test_case_2():
    ext = [DummyEntry("location", "weights.bin"), DummyEntry("offset", "128"), DummyEntry("length", "1024")]
    init = [DummyTensor("ext_tensor", external_data=ext)]
    return DummyModel(DummyGraph(init, []))


def generate_test_case_3():
    init = [DummyTensor("small", raw_data=b"abc")]
    return DummyModel(DummyGraph(init, []))
