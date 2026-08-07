import ref


def check(workdir):
    from tlog.memory import calculate_kv_memory
    from tlog.throughput import decode_throughput_ratio

    mem_res = calculate_kv_memory(num_layers=32, num_kv_heads=8, head_dim=128, max_seq_len=2048, dtype_bytes=2, total_memory=16 * 1024 * 1024 * 1024)
    want_mem = ref.ref_calculate_kv_memory(32, 8, 128, 2048, 2, 16 * 1024 * 1024 * 1024)

    out = {"memory_match": 0.0, "throughput_ratio_match": 0.0}

    if mem_res != want_mem:
        out["_note"] = f"memory result mismatch: got {mem_res}, want {want_mem}"
        return out
    out["memory_match"] = 1.0

    b1 = 120.5
    b64 = 4820.0
    ratio_got = decode_throughput_ratio(b1, b64)
    want_ratio = ref.ref_decode_throughput_ratio(b1, b64)

    if abs(ratio_got - want_ratio) > 1e-5:
        out["_note"] = f"throughput ratio got {ratio_got}, want {want_ratio}"
        return out
    out["throughput_ratio_match"] = 1.0
    return out
