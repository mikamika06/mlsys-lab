from aot_compare.profiling import compare_compilation_timings
from aot_compare.stablehlo_diff import parse_stablehlo_op_counts, diff_stablehlo_ops

class MockLowered:
    def __init__(self, name, ops):
        self.name = name
        self.ops = ops

    def compile(self):
        return lambda *args: 42

    def as_text(self, dialect="stablehlo"):
        return "\n".join([f'  "stablehlo.{op}"' for op in self.ops])

class MockJit:
    def __init__(self, ops_a=None, ops_b=None):
        self.ops_a = ops_a or ["dot", "add"]
        self.ops_b = ops_b or ["dot"]

    def fresh_instance(self):
        return MockJit(self.ops_a, self.ops_b)

    def lower(self, *args, flags=None):
        flags = flags or {}
        ops = self.ops_b if flags.get("opt") else self.ops_a
        return MockLowered("fn", ops)

    def __call__(self, *args):
        return 42

def test_compilation_profiling():
    jit_fn = MockJit()
    res = compare_compilation_timings(jit_fn, 1.0)
    assert "aot_compile_time" in res
    assert "jit_first_call_time" in res
    assert "jit_cached_time" in res
    assert "overhead_ratio" in res
    assert res["aot_compile_time"] >= 0.0
    assert res["jit_first_call_time"] >= 0.0

def test_stablehlo_op_diff():
    code = '  "stablehlo.dot" \n  "stablehlo.add" \n  "stablehlo.add"'
    counts = parse_stablehlo_op_counts(code)
    assert counts == {"dot": 1, "add": 2}

    jit_fn = MockJit(ops_a=["dot", "add", "add"], ops_b=["dot"])
    diff = diff_stablehlo_ops(jit_fn, (1.0,), flags_a={}, flags_b={"opt": True})
    assert diff["add"] == -2
    assert diff["dot"] == 0
