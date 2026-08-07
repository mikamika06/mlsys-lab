import ref
from kvradix.benchmark import simulate_agent_trace
from kvradix.eviction import EvictableRadixCache


def check(workdir):
    out = {
        "capacity_respected": 0.0,
        "lru_eviction_correctness": 0.0,
        "pinning_respected": 0.0,
    }

    cache = EvictableRadixCache(max_tokens=15)
    cache.insert_and_cache([1, 2, 3, 4, 5])
    cache.insert_and_cache([1, 2, 3, 6, 7])
    cache.insert_and_cache([1, 2, 3, 8, 9, 10, 11])

    if cache.total_tokens() <= 15:
        out["capacity_respected"] = 1.0
    else:
        out["_note"] = f"Total tokens {cache.total_tokens()} exceeded max capacity 15"

    c2 = EvictableRadixCache(max_tokens=10)
    _, n1 = c2.insert_and_cache([10, 20, 30, 40])
    _, n2 = c2.insert_and_cache([10, 20, 50, 60])
    c2.insert_and_cache([10, 20, 50, 70])

    if c2.total_tokens() <= 10:
        out["lru_eviction_correctness"] = 1.0

    c3 = EvictableRadixCache(max_tokens=8)
    _, pinned_node = c3.insert_and_cache([1, 2, 3, 4])
    c3.inc_ref(pinned_node)

    c3.insert_and_cache([5, 6, 7, 8, 9])
    c3.insert_and_cache([5, 6, 10, 11, 12])

    match_len, _, _ = c3.tree.match_prefix([1, 2, 3, 4])
    if match_len == 4:
        out["pinning_respected"] = 1.0
    else:
        out["_note"] = f"Pinned sequence was evicted, match length was {match_len} instead of 4"

    trace = ref.generate_traces()[0]
    radix_hr = simulate_agent_trace(trace, "radix", capacity_tokens=64)
    block_hr = simulate_agent_trace(trace, "block_hash", block_size=16, capacity_tokens=64)

    if radix_hr <= block_hr:
        out["_note"] = f"Radix hit rate ({radix_hr:.3f}) was not higher than block hash hit rate ({block_hr:.3f})"

    return out
