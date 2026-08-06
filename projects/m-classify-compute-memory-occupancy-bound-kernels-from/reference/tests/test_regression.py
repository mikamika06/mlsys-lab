from profiler.analysis import parse_torch_trace_kernel


def test_tflops_calculation():
    sample = {
        "dur": 1000.0,
        "args": {
            "block X": 128,
            "block Y": 1,
            "block Z": 1,
            "grid X": 64,
            "grid Y": 1,
            "grid Z": 1,
            "flops": 1000000000000.0,
        }
    }
    res = parse_torch_trace_kernel(sample)
    assert res["tflops"] == 1.0
    assert res["block_size"] == [128, 1, 1]
    assert res["grid_size"] == [64, 1, 1]
