import ref


def check(workdir):
    from distmem.memory import compute_memory_comparison

    ok = 0
    for case in ref.MEMORY_CASES:
        want_ddp = ref.ref_ddp_memory(
            case["params"],
            case["bytes_per_param"],
            case["optimizer_bytes_per_param"],
        )
        want_fsdp = ref.ref_fsdp_memory(
            case["params"],
            case["bytes_per_param"],
            case["optimizer_bytes_per_param"],
            case["world_size"],
        )
        got_ddp, got_fsdp = compute_memory_comparison(
            case["params"],
            case["bytes_per_param"],
            case["optimizer_bytes_per_param"],
            case["world_size"],
        )
        if (
            abs(got_ddp - want_ddp) < 1e-5
            and abs(got_fsdp - want_fsdp) < 1e-5
        ):
            ok += 1
    return {"memory_matched": 1.0 if ok == len(ref.MEMORY_CASES) else 0.0}
