import sys

sys.path.insert(0, ".")
from transform import insert_cast_nodes


def test_island_cast_injection_order():
    graph = [
        {"name": "A", "op": "linear"},
        {"name": "B", "op": "exp"},
        {"name": "C", "op": "normalize"}
    ]
    new_g = insert_cast_nodes(graph, "B", "C")
    assert len(new_g) == 5
    assert new_g[1]["op"] == "cast"
    assert new_g[1]["to"] == "float32"
    assert new_g[2]["name"] == "B"
    assert new_g[3]["name"] == "C"
    assert new_g[4]["op"] == "cast"
    assert new_g[4]["to"] == "float16"
