import struct
from arena.planner import plan_activation_arena
from arena.pte_reader import parse_pte_constants


def test_greedy_ordering_differs_from_input_order():
    buffers = [
        {"id": "small", "size": 100, "liveness": (0, 10)},
        {"id": "big1", "size": 1000, "liveness": (0, 4)},
        {"id": "big2", "size": 1000, "liveness": (5, 10)},
    ]
    res = plan_activation_arena(buffers, default_alignment=64)
    assert res["offsets"]["big1"] == 0
    assert res["offsets"]["big2"] == 0
    assert res["offsets"]["small"] == 1024


def test_pte_constants():
    header = struct.pack("<4sII", b"PTE1", 1, 1)
    seg = struct.pack("<IQQI", 1, 4096, 2048, 64)
    data = header + seg
    res = parse_pte_constants(data)
    assert res == {"offset": 4096, "size": 2048, "alignment": 64}
