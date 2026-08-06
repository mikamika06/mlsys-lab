import sys
sys.path.insert(0, ".")
from onnxtools.analyzer import find_initializer_inputs, resolve_external_ranges, predict_2gb_ceiling


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


def test_find_initializer_inputs():
    init = [DummyTensor("weight1"), DummyTensor("weight2")]
    inp = [DummyValueInfo("weight1"), DummyValueInfo("data")]
    g = DummyGraph(init, inp)
    model = DummyModel(g)
    res = find_initializer_inputs(model)
    assert res == ["weight1"]


def test_predict_2gb_ceiling():
    big_data = b"x" * (2 * 1024 * 1024 * 1024 + 10)
    init = [DummyTensor("huge", raw_data=big_data)]
    g = DummyGraph(init, [])
    model = DummyModel(g)
    assert predict_2gb_ceiling(model) is True
