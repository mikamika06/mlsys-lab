def expected_ulysses_comm(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    world_size: int,
    bytes_per_elem: int = 2,
) -> dict:
    """Calculates expected communication stats for Ulysses sequence parallelism."""
    total_elems = batch_size * seq_len * num_heads * head_dim
    total_bytes = total_elems * bytes_per_elem
    chunk_bytes = total_bytes // (world_size * world_size)
    send_bytes_per_rank = (world_size - 1) * chunk_bytes
    total_network_bytes = world_size * send_bytes_per_rank
    return {
        "chunk_bytes": chunk_bytes,
        "send_bytes_per_rank": send_bytes_per_rank,
        "total_network_bytes": total_network_bytes,
    }


def verify_comm_log(records: list[dict], config: dict) -> dict:
    """Verifies recorded DeepSpeed all-to-all communication log entries."""
    b = config["batch_size"]
    s = config["seq_len"]
    h = config["num_heads"]
    d = config["head_dim"]
    p = config["world_size"]
    e = config.get("bytes_per_elem", 2)

    exp = expected_ulysses_comm(s, h, d, b, p, e)
    expected_send = exp["send_bytes_per_rank"]

    mismatches = 0
    max_rel_err = 0.0

    for rec in records:
        sb = rec.get("send_bytes", 0)
        rb = rec.get("recv_bytes", 0)
        err_s = abs(sb - expected_send) / max(1.0, float(expected_send))
        err_r = abs(rb - expected_send) / max(1.0, float(expected_send))
        err = max(err_s, err_r)
        if err > max_rel_err:
            max_rel_err = err
        if err > 1e-4:
            mismatches += 1

    return {
        "valid": mismatches == 0,
        "mismatches": mismatches,
        "max_rel_err": float(max_rel_err),
        "expected_send_bytes": expected_send,
    }
