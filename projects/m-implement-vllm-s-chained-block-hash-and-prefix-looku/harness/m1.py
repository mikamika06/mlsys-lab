import ref


def check(workdir):
    from prefix_cache.hash import build_prefix_hash_chain, compute_block_hash
    from prefix_cache.lookup import PrefixCacheManager

    out = {"hashes_match": 0.0, "lookups_match": 0.0}

    hash_ok = True
    for seq in ref.TEST_TOKEN_SEQS:
        for bsize in [2, 4]:
            ref_hashes = ref.build_prefix_hash_chain(seq, bsize)
            try:
                got_hashes = build_prefix_hash_chain(seq, bsize)
            except Exception as e:
                out["_note"] = f"build_prefix_hash_chain failed: {e}"
                return out
            if got_hashes != ref_hashes:
                hash_ok = False
                out["_note"] = (
                    f"Hash mismatch for seq length {len(seq)} block_size {bsize}"
                )
                break
        if not hash_ok:
            break

    if hash_ok:
        out["hashes_match"] = 1.0

    lookup_ok = True
    for bsize in [2, 4]:
        ref_mgr = ref.PrefixCacheManager(bsize)
        user_mgr = PrefixCacheManager(bsize)

        for seq in ref.TEST_TOKEN_SEQS:
            ref_b, ref_t = ref_mgr.lookup_prefix(seq)
            try:
                user_b, user_t = user_mgr.lookup_prefix(seq)
            except Exception as e:
                out["_note"] = f"lookup_prefix failed: {e}"
                return out

            if (ref_b, ref_t) != (user_b, user_t):
                lookup_ok = False
                out["_note"] = f"Lookup mismatch before insert: got {(user_b, user_t)}, want {(ref_b, ref_t)}"
                break

            ref_mgr.insert_sequence(seq)
            user_mgr.insert_sequence(seq)

            ref_b2, ref_t2 = ref_mgr.lookup_prefix(seq)
            user_b2, user_t2 = user_mgr.lookup_prefix(seq)
            if (ref_b2, ref_t2) != (user_b2, user_t2):
                lookup_ok = False
                out["_note"] = f"Lookup mismatch after insert: got {(user_b2, user_t2)}, want {(ref_b2, ref_t2)}"
                break

        if not lookup_ok:
            break

    if lookup_ok:
        out["lookups_match"] = 1.0

    return out
