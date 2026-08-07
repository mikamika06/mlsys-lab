import ref


def check(workdir):
    from kvblock.cache import PrefixCacheSimulator
    from kvblock.triage import triage_block_table

    out = {"sim_matches": 0.0, "triage_matches": 0.0}

    requests = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 10, 11, 12, 13],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [20, 21, 22, 23, 24, 25, 26, 27],
        [1, 2, 3, 4, 30, 31, 32, 33],
    ]

    sim = PrefixCacheSimulator(block_size=4, max_blocks=3)
    for req in requests:
        sim.process_request(req)

    got_rate = sim.hit_rate()
    _, _, want_rate = ref.simulate_prefix_cache(requests, block_size=4, max_blocks=3)

    if abs(got_rate - want_rate) < 1e-6:
        out["sim_matches"] = 1.0
    else:
        out["_note"] = f"cache sim hit_rate mismatch: got {got_rate}, want {want_rate}"

    corrupt_tables = [
        ([0, 1, 2], 48, 16, 100),
        ([0, 1, 999, 1], 64, 16, 100),
        ([0, 1], 64, 16, 100),
        ([5, 10, 15, 20, 25], 32, 16, 50),
    ]

    triage_ok = True
    for bt, seq_len, blk_size, max_id in corrupt_tables:
        got = triage_block_table(bt, seq_len, blk_size, max_id)
        want = ref.reference_triage(bt, seq_len, blk_size, max_id)
        if got != want:
            triage_ok = False
            out["_note"] = f"triage mismatch: got {got}, want {want}"
            break

    if triage_ok:
        out["triage_matches"] = 1.0

    return out
