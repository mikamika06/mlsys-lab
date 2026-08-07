import ref


def check(workdir):
    from cache.hash import compute_block_hash

    out = {"hashes_distinct": 0.0}
    try:
        h_txt = compute_block_hash(b"test_payload", is_image=False, truncate_bits=32)
        h_img = compute_block_hash(b"test_payload", is_image=True, truncate_bits=32)
        if h_txt != h_img:
            out["hashes_distinct"] = 1.0
        else:
            out["_note"] = "image prompt hash collided with text-only block hash"
    except Exception as e:
        out["_note"] = f"exception during hash check: {type(e).__name__}: {str(e)[:100]}"
    return out
