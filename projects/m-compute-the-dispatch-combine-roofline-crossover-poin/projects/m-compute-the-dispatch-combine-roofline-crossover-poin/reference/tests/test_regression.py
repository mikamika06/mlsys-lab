from moeroof.crossover import compute_crossover
from moeroof.packing import pack_experts


def test_crossover_valid():
    val = compute_crossover(4096, 16, 200.0, 300.0)
    assert isinstance(val, int)
    assert val > 0


def test_packing_valid():
    loads = [10, 20, 30, 40]
    gpus = pack_experts(loads, 2)
    assert len(gpus) == 2
    total_allocated = sum(len(g) for g in gpus)
    assert total_allocated == 4
