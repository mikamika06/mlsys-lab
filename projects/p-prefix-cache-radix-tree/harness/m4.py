def check(workdir):
    try:
        from prefix_cache.cache import PrefixCache
    except ImportError:
        return {"hit_rate_ok": 0.0}

    c = PrefixCache(block_size=16)

    shared = [(i, i+1) for i in range(5)]
    hits = 0

    next_id = 0
    active = []

    try:
        for req in range(100):
            blocks = shared + [(req,)]
            matched = c.match("t1", blocks)
            hits += len(matched)

            unmatched = blocks[len(matched):]
            if unmatched:
                new_ids = list(range(next_id, next_id + len(unmatched)))
                next_id += len(unmatched)
                c.insert("t1", blocks, matched + new_ids)
                c.inc_ref(matched + new_ids)
                active.append(matched + new_ids)
            else:
                c.inc_ref(matched)
                active.append(matched)

            if len(active) > 10:
                oldest = active.pop(0)
                c.dec_ref(oldest)

            while len(c.nodes) > 50:
                if c.evict() is None:
                    break

        return {"hit_rate_ok": 1.0 if hits >= 400 else 0.0}
    except Exception:
        return {"hit_rate_ok": 0.0}
