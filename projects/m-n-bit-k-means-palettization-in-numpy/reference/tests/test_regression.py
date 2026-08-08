import sys

sys.path.insert(0, ".")
from palettize.compress import palettize_size_bytes


def test_palette_size_is_accounted_for():
    assert palettize_size_bytes(4096, 4, 1) == 2112
    assert palettize_size_bytes(4096, 8, 2) == 4096
