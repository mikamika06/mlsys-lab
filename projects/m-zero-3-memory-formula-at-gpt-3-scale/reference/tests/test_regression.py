import sys
sys.path.insert(0, ".")
from zerothree.memory import calculate_zero3_memory
from zerothree.schedule import simulate_all_gather_free_cycle
from zerothree.volume import calculate_communication_volume

def test_memory_scaling():
    mem = calculate_zero3_memory(175_000_000_000, 4, 64)
    assert mem > 0
    assert isinstance(mem, float)

def test_schedule_peak():
    res = simulate_all_gather_free_cycle([1024, 2048, 512], 8)
    assert "peak_memory" in res
    assert res["peak_memory"] > 0

def test_communication_volume_multiplier():
    vol = calculate_communication_volume(1_000_000, 2, 8)
    psi = 2.0 * 1_000_000
    expected = 3.0 * psi * (7.0 / 8.0)
    assert abs(vol - expected) < 1e-5
