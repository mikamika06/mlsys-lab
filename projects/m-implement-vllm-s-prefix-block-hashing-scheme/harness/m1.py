import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from prefix_cache.block_hash import compute_prefix_hashes, hash_block

    out = {"blocks_matched": 0.0}

    ref_hashes = ref.compute_prefix_hashes(ref.DATASET[0], ref.BLOCK_SIZE)
    got_hashes = compute_prefix_hashes(ref.DATASET[0], ref.BLOCK_SIZE)

    if ref_hashes == got_hashes:
        out["blocks_matched"] = 1.0
    else:
        out["_note"] = f"Expected hashes {ref_hashes[:2]}, got {got_hashes[:2]}"

    return out
