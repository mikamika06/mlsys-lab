from brgemm.dispatch import reconstruct_call_sequence
from brgemm.roofline import analyze_roofline, compute_memory_traffic


def test_dispatch_sequence():
    calls = reconstruct_call_sequence(128, 128, 128, 64, 64, 32)
    assert len(calls) == 4
    for call in calls:
        assert call["batch_size"] == 4
        assert len(call["a_offsets"]) == 4
        assert len(call["b_offsets"]) == 4


def test_roofline_memory_traffic():
    traffic = compute_memory_traffic(128, 128, 128, 64, 64, 32, bytes_per_elem=2)
    assert traffic == (128 * 128 * 2 * 2 + 128 * 128 * 2 * 2 + 2 * 128 * 128 * 2)


def test_roofline_analysis():
    res = analyze_roofline(1024, 1024, 1024, 64, 64, 32, 1000.0, 50.0, 5.0)
    assert res["flops"] == 2 * 1024 * 1024 * 1024
    assert res["dram_bytes"] > 0
    assert 0.0 < res["efficiency"] <= 1.0
