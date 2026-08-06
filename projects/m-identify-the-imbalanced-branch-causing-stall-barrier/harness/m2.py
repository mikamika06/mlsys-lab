import ref

def check(workdir):
    from stallprof.verifier import verify_sync_removal
    out = {"verification_matched": 0.0}
    base = ref.KERNELS[0]
    mod = {
        "kernel_id": base["kernel_id"],
        "warp_stats": {"stall_barrier": 2000, "stall_mio": 1000, "stall_not_selected": 500, "active": 500}
    }
    speedup = 1.30
    want = ref.verify_sync_removal(base, mod, speedup)
    got = verify_sync_removal(base, mod, speedup)
    if got == want:
        out["verification_matched"] = 1.0
    else:
        out["_note"] = f"verification result mismatch: got {got}, want {want}"
    return out
