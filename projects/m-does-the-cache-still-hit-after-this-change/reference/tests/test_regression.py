from cacheutils.hasher import stable_hash


def test_stable_hash_ignores_volatile():
    p1 = {"model": "llama", "lr": 0.001, "_volatile_time": 123}
    p2 = {"model": "llama", "lr": 0.001, "_volatile_time": 456}
    assert stable_hash(p1) == stable_hash(p2)


def test_stable_hash_order_independent():
    p1 = {"b": 2, "a": 1}
    p2 = {"a": 1, "b": 2}
    assert stable_hash(p1) == stable_hash(p2)
