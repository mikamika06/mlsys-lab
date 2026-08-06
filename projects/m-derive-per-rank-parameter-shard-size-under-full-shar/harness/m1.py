import ref

def check(workdir):
    from fsdp_model.shard import compute_shard_sizes
    cases = ref.get_test_cases()
    matched = 0
    for total_params, world_size in cases:
        base = total_params // world_size
        rem = total_params % world_size
        want = [base + (1 if i < rem else 0) for i in range(world_size)]
        try:
            got = compute_shard_sizes(total_params, world_size)
            if got == want:
                matched += 1
        except Exception:
            pass
    success = 1.0 if matched == len(cases) else 0.0
    out = {"shard_sizes_matched": success}
    if success == 0.0:
        out["_note"] = f"Failed shard size check on cases out of {len(cases)}"
    return out
