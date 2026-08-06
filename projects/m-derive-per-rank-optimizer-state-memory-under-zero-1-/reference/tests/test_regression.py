from zeromem.optimizer import optimizer_state_memory
from zeromem.stages import total_memory
from zeromem.budget import min_zero_stage


def test_optimizer_state_memory_basic():
    assert optimizer_state_memory(1000, 4, True) == 3000
    assert optimizer_state_memory(1000, 4, False) == 12000


def test_total_memory_stages():
    m1 = total_memory(1000, 4, 1, 100)
    m2 = total_memory(1000, 4, 2, 100)
    m3 = total_memory(1000, 4, 3, 100)
    assert m1 > m2
    assert m2 > m3


def test_min_zero_stage_selection():
    stage = min_zero_stage(1000, 4, 50000, 100)
    assert stage in (1, 2, 3)
