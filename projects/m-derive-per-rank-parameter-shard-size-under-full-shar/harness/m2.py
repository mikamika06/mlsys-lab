import ref

def check(workdir):
    from fsdp_model.comm import compute_communication_volume
    cases = ref.get_comm_cases()
    matched = 0
    for num_params, bytes_per_param, strategy, world_size in cases:
        total_bytes = num_params * bytes_per_param
        if strategy == "FULL_SHARD":
            want = int(2 * total_bytes * (world_size - 1) / world_size)
        elif strategy == "SHARD_GRAD_OP":
            want = int(total_bytes * (world_size - 1) / world_size)
        elif strategy == "NO_SHARD":
            want = 0
        else:
            want = -1

        try:
            got = compute_communication_volume(num_params, bytes_per_param, strategy, world_size)
            if got == want:
                matched += 1
        except Exception:
            pass
    success = 1.0 if matched == len(cases) else 0.0
    out = {"comm_volumes_matched": success}
    if success == 0.0:
        out["_note"] = f"Communication volume mismatch across test cases"
    return out
