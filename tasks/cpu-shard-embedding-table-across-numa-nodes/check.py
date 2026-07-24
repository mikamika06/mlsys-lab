import numpy as np
from mlsys.sim import cache as cachesim


def _reference_shard(num_embeddings: int, dim: int, num_nodes: int):
    """Equal contiguous shard reference."""
    base = num_embeddings // num_nodes
    rem = num_embeddings % num_nodes
    shards = []
    start = 0
    for i in range(num_nodes):
        extra = 1 if i < rem else 0
        end = start + base + extra
        shards.append((start, end))
        start = end
    return shards


def _simulate_accesses(shards, num_embeddings, dim, num_nodes, batch=128):
    """Model each node accessing its own shard indices uniformly."""
    accesses = []
    for node_id, (s, e) in enumerate(shards):
        idxs = np.random.default_rng(42 + node_id).integers(s, e, size=batch, dtype=np.int64)
        # base byte address for each row, each element is 8 bytes (float64)
        addrs = idxs * dim * 8
        accesses.append(addrs)
    # Flatten all nodes' traces in order
    all_addrs = np.concatenate(accesses)
    # Simulate with small fixed cache
    res = cachesim.simulate(all_addrs, line_bytes=64, sets=64, ways=8)
    return res["misses"]


def grade(sol, fx) -> dict:
    np.random.seed(0)
    num_embeddings, dim, num_nodes = 10000, 64, 4
    ref_shard = _reference_shard(num_embeddings, dim, num_nodes)
    try:
        cand_shard = sol.shard_embedding_table(num_embeddings, dim, num_nodes)
    except Exception:
        return {"exact_match": 0.0}
    ref_miss = _simulate_accesses(ref_shard, num_embeddings, dim, num_nodes)
    cand_miss = _simulate_accesses(cand_shard, num_embeddings, dim, num_nodes)
    ok = float(ref_miss == cand_miss)
    return {"exact_match": ok}
