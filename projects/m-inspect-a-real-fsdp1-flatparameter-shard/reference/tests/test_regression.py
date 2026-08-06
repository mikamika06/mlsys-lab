import sys
sys.path.insert(0, ".")
from fsdp_shard.inspect import inspect_shard
from fsdp_shard.padding import compute_padding_overhead
from fsdp_shard.balance import per_rank_balance

def test_shard_bounds_cover_total():
    params = [100, 200, 300]
    ws = 2
    shards = [inspect_shard(params, ws, r) for r in range(ws)]
    assert shards[0]["start"] == 0
    assert shards[-1]["end"] == shards[0]["shard_size"] * ws

def test_padding_ratio_bounds():
    params = [15, 20]
    ws = 4
    ratio = compute_padding_overhead(params, ws)
    assert 0.0 <= ratio < 1.0

def test_per_rank_balance_sum():
    params = [1024, 512, 256]
    ws = 4
    bal = per_rank_balance(params, ws)
    assert len(bal) == ws
    assert sum(bal) >= sum(params)
    assert all(x == bal[0] for x in bal)
