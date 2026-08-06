from recompiles.simulate import simulate_lru_cache

def test_lru_basic():
    shapes = [(1, 128), (2, 128), (1, 128)]
    res = simulate_lru_cache(shapes, capacity=2)
    assert res == 2

def test_lru_eviction():
    shapes = [(1, 128), (2, 128), (3, 128), (1, 128)]
    res = simulate_lru_cache(shapes, capacity=2)
    assert res == 4
