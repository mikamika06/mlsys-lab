import sys
sys.path.insert(0, ".")
from zeroproj.memory import compute_memory_table
from zeroproj.partition import assign_partitions
from zeroproj.binpack import bin_pack_partition

def test_memory_table_structure():
    table = compute_memory_table([100, 200, 300], 2, dtype_bytes=4)
    assert "plain" in table
    assert "zero1" in table
    assert table["zero1"]["optimizer_states"] < table["plain"]["optimizer_states"]

def test_partition_balance():
    sizes = [10, 20, 30, 40, 50]
    parts = assign_partitions(sizes, 2)
    assert len(parts) == 2
    assert sum(sizes[i] for i in parts[0]) > 0

def test_bin_pack_valid():
    sizes = [10, 10, 10, 10]
    parts = bin_pack_partition(sizes, 2)
    assert len(parts) == 2
    seen = [i for p in parts for i in p]
    assert sorted(seen) == [0, 1, 2, 3]
