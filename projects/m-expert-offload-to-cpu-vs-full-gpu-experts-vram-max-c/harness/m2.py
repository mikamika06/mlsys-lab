import ref


def check(workdir):
    from moeoffload.oom import diagnose_oom, safe_allocation_limit

    out = {"diagnoses_matched": 0.0, "fixes_applied": 0.0}
    test_cases = [
        ({"flash_attn": True, "cpu_offload_experts": True, "batch_size": 64}, 8000, 10000),
        ({"flash_attn": False, "cpu_offload_experts": False, "tensor_parallel": 2, "pinned_buffers": False}, 12000, 10000),
        ({"flash_attn": False, "cpu_offload_experts": True}, 16000, 15000)
    ]

    ok_diag = 0
    for flags, avail, req in test_cases:
        want = ref.diagnose_oom(flags, avail, req)
        got = diagnose_oom(flags, avail, req)
        if sorted(want) == sorted(got):
            ok_diag += 1

    out["diagnoses_matched"] = float(ok_diag)

    test_flags = {"flash_attn": True, "cpu_offload_experts": True}
    vram_tot = 16384
    want_limit = ref.safe_allocation_limit(test_flags, vram_tot)
    got_limit = safe_allocation_limit(test_flags, vram_tot)

    if want_limit == got_limit:
        out["fixes_applied"] = 1.0
    else:
        out["_note"] = f"safe_allocation_limit got {got_limit}, want {want_limit}"

    return out
