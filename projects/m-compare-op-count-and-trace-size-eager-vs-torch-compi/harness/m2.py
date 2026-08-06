import ref

def check(workdir):
    from prof.cache import compute_cache_hit_ratio
    _, sequences = ref.generate_fixtures()
    ok = 0
    out = {"ratio_matched": 0.0, "total": float(len(sequences))}
    for i, seq in enumerate(sequences):
        got = compute_cache_hit_ratio(seq, cache_capacity=4)
        seen = []
        hits = 0
        misses = 0
        for b in seq:
            if b in seen:
                hits += 1
            else:
                misses += 1
                seen.append(b)
                if len(seen) > 4:
                    seen.pop(0)
        total = hits + misses
        want = hits / total if total > 0 else 0.0
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cache ratio mismatch at index {i}"
    out["ratio_matched"] = 1.0 if ok == len(sequences) else 0.0
    return out
