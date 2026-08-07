import sys
sys.path.insert(0, ".")

from footprint.split import analyze_three_way_footprint
from footprint.selective import selective_registration_win
from footprint.predictor import predict_peak_rss


def test_three_way_split_sum():
    binary = {"text_bytes": 1000, "rodata_bytes": 500, "data_bytes": 100, "bss_bytes": 200}
    runtime = {"base_heap_bytes": 2000, "context_struct_bytes": 300, "arena_metadata_bytes": 100}
    tensors = [
        {"numel": 100, "elem_bytes": 4, "category": "weight"},
        {"numel": 50, "elem_bytes": 4, "category": "activation"}
    ]
    res = analyze_three_way_footprint(binary, runtime, tensors)
    expected_dynamic = 400 + 200
    expected_static = 1800
    expected_runtime = 2400
    assert res["static_binary_bytes"] == expected_static
    assert res["runtime_infra_bytes"] == expected_runtime
    assert res["dynamic_tensors_bytes"] == expected_dynamic
    assert res["total_footprint_bytes"] == expected_static + expected_runtime + expected_dynamic


def test_selective_registration_savings():
    all_kernels = [
        {"op": "conv2d", "code_bytes": 5000, "table_bytes": 200},
        {"op": "relu", "code_bytes": 1000, "table_bytes": 50},
        {"op": "reshape", "code_bytes": 800, "table_bytes": 40}
    ]
    used_ops = ["conv2d", "relu"]
    res = selective_registration_win(all_kernels, used_ops)
    assert res["binary_bytes_saved"] == 800
    assert res["table_bytes_saved"] == 40
    assert res["total_bytes_saved"] == 840
    assert "reshape" in res["pruned_ops"]


def test_peak_rss_monotonic_or_bounded():
    plan = [
        {"name": "t1", "start_step": 0, "end_step": 2, "size_bytes": 100},
        {"name": "t2", "start_step": 1, "end_step": 3, "size_bytes": 200},
    ]
    res = predict_peak_rss(plan, alignment=64, overhead_bytes=50)
    assert res["peak_rss_bytes"] == 50 + 128 + 256
    assert res["peak_step"] == 1
    assert res["active_tensors_at_peak"] == ["t1", "t2"]
