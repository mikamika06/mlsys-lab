import sys

sys.path.insert(0, ".")
from bench.measure import measure_warmup_bias


class DummyEngine:
    def __init__(self):
        self.runs = 0

    def generate(self, prompt_len, dtype):
        self.runs += 1
        if self.runs <= 2:
            return 100.0
        return 10.0


def test_warmup_is_discarded():
    engine = DummyEngine()
    steady, bias = measure_warmup_bias(engine, 10, 2, 3, "fp16")
    assert steady == 10.0
    assert bias == 36.0
