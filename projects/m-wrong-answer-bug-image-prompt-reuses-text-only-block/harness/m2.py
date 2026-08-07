import ref


def check(workdir):
    from cache.hash import search_collision

    out = {"collisions_resolved": 0.0}
    try:
        colls = search_collision(ref.TEST_BLOCKS, truncate_bits=32)
        if isinstance(colls, list):
            out["collisions_resolved"] = 1.0
        else:
            out["_note"] = "search_collision did not return a list"
    except Exception as e:
        out["_note"] = f"exception during collision search: {type(e).__name__}: {str(e)[:100]}"
    return out
