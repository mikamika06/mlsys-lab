import ref


def check(workdir):
    from fsdpshards.padding import compare_fsdp1_fsdp2_chunking

    out = {"chunking_matched": 0.0, "padding_diff_matched": 0.0}
    total = len(ref.M2_CASES)
    c_ok = 0
    p_ok = 0
    for i, (shape, mesh_size, dtype_bytes) in enumerate(ref.M2_CASES):
        want = ref.compare_fsdp1_fsdp2_chunking(shape, mesh_size, dtype_bytes)
        got = compare_fsdp1_fsdp2_chunking(shape, mesh_size, dtype_bytes)

        if got.get("fsdp2_local_shapes") == want["fsdp2_local_shapes"]:
            c_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i} chunking mismatch: got {got.get('fsdp2_local_shapes')}, want {want['fsdp2_local_shapes']}"

        if (
            got.get("fsdp1_wasted_bytes") == want["fsdp1_wasted_bytes"]
            and got.get("fsdp2_wasted_bytes") == want["fsdp2_wasted_bytes"]
            and got.get("fsdp1_total_bytes") == want["fsdp1_total_bytes"]
            and got.get("fsdp2_total_bytes") == want["fsdp2_total_bytes"]
        ):
            p_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i} padding bytes mismatch: got {got}, want {want}"

    if c_ok == total:
        out["chunking_matched"] = 1.0
    if p_ok == total:
        out["padding_diff_matched"] = 1.0

    return out
