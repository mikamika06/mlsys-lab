import fsdp_model.memory as mem

def test_transient_peak_memory_basic():
    val = mem.compute_transient_peak_memory(1000, 200, 4)
    assert val > 0
