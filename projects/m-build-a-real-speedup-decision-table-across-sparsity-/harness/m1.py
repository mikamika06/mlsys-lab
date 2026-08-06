import ref


def check(workdir):
    from sparsity.storage import (
        compute_theoretical_bytes,
        dense_pt_saved_bytes,
        compute_csr_breakeven_sparsity,
    )

    out = {
        "theoretical_match": 1.0,
        "torch_dense_saved_match": 1.0,
        "breakeven_match": 1.0,
    }

    formats = ["dense", "csr", "coo", "2:4"]

    for cfg in ref.CONFIGS:
        shape = cfg["shape"]
        bits = cfg["dtype_bits"]
        sp = cfg["sparsity"]

        for fmt in formats:
            want = ref.ref_compute_theoretical_bytes(shape, bits, fmt, sp)
            got = compute_theoretical_bytes(shape, bits, fmt, sp)
            if abs(want - got) / max(1.0, want) > 1e-4:
                out["theoretical_match"] = 0.0
                out["_note"] = f"Theoretical bytes mismatch for {fmt}: want {want}, got {got}"
                return out

        want_pt = ref.ref_dense_pt_saved_bytes(shape, bits, sp)
        got_pt = dense_pt_saved_bytes(shape, bits, sp)
        if abs(want_pt - got_pt) > 1e-4:
            out["torch_dense_saved_match"] = 0.0
            out["_note"] = f"Dense PT saved bytes mismatch: want {want_pt}, got {got_pt}"
            return out

        want_be = ref.ref_compute_csr_breakeven_sparsity(shape, bits)
        got_be = compute_csr_breakeven_sparsity(shape, bits)
        if abs(want_be - got_be) > 1e-4:
            out["breakeven_match"] = 0.0
            out["_note"] = f"Breakeven sparsity mismatch: want {want_be}, got {got_be}"
            return out

    return out
