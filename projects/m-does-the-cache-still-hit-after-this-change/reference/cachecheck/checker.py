from cachecheck.forensics import analyze_cache_dir
from cachecheck.hasher import stable_hash


def check_cache_hit(state_a, state_b, cache_dir):
    h_a = stable_hash(state_a)
    h_b = stable_hash(state_b)
    forensics = analyze_cache_dir(cache_dir)
    hit = (h_a == h_b)
    return {"hit": hit, "hash_a": h_a, "hash_b": h_b, "forensics": forensics}
