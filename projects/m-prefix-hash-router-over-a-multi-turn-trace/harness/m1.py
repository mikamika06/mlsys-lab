import sys


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from router.prefix import (
            block_hash,
            tokenize_into_blocks,
            compute_prefix_match,
            PrefixRouter,
        )
        import ref

        mismatches = 0
        total_checks = 0

        tokens_sample = [1, 2, 3, 4, 5, 6, 7, 8]
        if block_hash(tokens_sample) != ref.block_hash(tokens_sample):
            mismatches += 1
        total_checks += 1

        b_got = tokenize_into_blocks(tokens_sample, 4)
        b_ref = ref.tokenize_into_blocks(tokens_sample, 4)
        if b_got != b_ref:
            mismatches += 1
        total_checks += 1

        if compute_prefix_match(b_got, b_ref) != ref.compute_prefix_match(b_got, b_ref):
            mismatches += 1
        total_checks += 1

        for trace in ref.TRACES:
            router_got = PrefixRouter(
                num_workers=4, max_blocks_per_worker=16, block_size=4
            )
            router_ref = ref.PrefixRouter(
                num_workers=4, max_blocks_per_worker=16, block_size=4
            )
            for req in trace:
                toks = req["tokens"]
                w_got, m_got = router_got.route(toks)
                w_ref, m_ref = router_ref.route(toks)
                if w_got != w_ref or m_got != m_ref:
                    mismatches += 1
                total_checks += 1
                router_got.update_cache(w_got, toks)
                router_ref.update_cache(w_ref, toks)

        rel_err = float(mismatches) / float(total_checks) if total_checks > 0 else 0.0
        return {"rel_err": rel_err}
    except Exception:
        return {"rel_err": 1.0}
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)
