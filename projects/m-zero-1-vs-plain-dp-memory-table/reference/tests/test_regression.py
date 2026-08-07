import sys

sys.path.insert(0, ".")
from zerodp.memory import calc_memory_table
from zerodp.partition import partition_bin_packing, partition_flat_contiguous


def test_bin_packing_minimizes_max_load():
    sizes = [10, 50, 30, 20, 40]
    res = partition_bin_packing(sizes, 2)
    assert res["max_load"] == 80
    assert res["imbalance"] == 10


def test_bin_packing_structure():
    sizes = [100, 200, 300, 400]
    res = partition_bin_packing(sizes, 2)
    assert sum(res["loads"]) == sum(sizes)
    all_assigned = [idx for rank in res["assignments"] for idx in rank]
    assert sorted(all_assigned) == list(range(len(sizes)))


def test_memory_table_savings():
    res = calc_memory_table([1000, 2000], 4)
    assert res["total_savings_bytes"] > 0
    assert res["zero1"]["opt_bytes"] == res["plain_dp"]["opt_bytes"] // 4


def test_flat_partition_alignment():
    res = partition_flat_contiguous([100, 200], 2, alignment=8)
    assert res["per_rank_size"] % 8 == 0
