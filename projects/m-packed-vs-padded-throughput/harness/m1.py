import ref

def check(workdir):
    try:
        from seqpack.core import build_cu_seqlens, packed_cost
    except ImportError:
        return {"cu_seqlens_matched": 0.0, "packed_cost_matched": 0.0}

    out = {"cu_seqlens_matched": 0.0, "packed_cost_matched": 0.0}
    cu_ok = 0
    pc_ok = 0

    for seqlens, block_size, _ in ref.FIXTURES:
        try:
            got_cu = build_cu_seqlens(seqlens)
            want_cu = ref.build_cu_seqlens(seqlens)
            if got_cu == want_cu:
                cu_ok += 1
        except Exception:
            pass

        try:
            got_pc = packed_cost(seqlens, block_size)
            want_pc = ref.packed_cost(seqlens, block_size)
            if got_pc == want_pc:
                pc_ok += 1
        except Exception:
            pass

    out["cu_seqlens_matched"] = float(cu_ok) / len(ref.FIXTURES)
    out["packed_cost_matched"] = float(pc_ok) / len(ref.FIXTURES)
    return out
