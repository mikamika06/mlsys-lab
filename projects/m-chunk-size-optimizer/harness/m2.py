import ref

def check(workdir):
    from optimizer.chunking import benchmark_serving_frontier, optimize_chunk_size

    trace = ref.generate_trace()
    sizes = [4, 8, 16, 32]
    out = {"frontier_matched": 0.0, "argmin_index": 0.0}
    ok_front = 0

    for s in sizes:
        sv = ref.calculate_prefix_savings(trace, s)
        want_f = ref.benchmark_serving_frontier(trace, s, sv)
        try:
            got_f = benchmark_serving_frontier(trace, s, sv)
            if abs(got_f - want_f) < 1e-5:
                ok_front += 1
            else:
                out["_note_front"] = f"size {s}: got {got_f}, want {want_f}"
        except Exception:
            pass

    out["frontier_matched"] = float(ok_front)

    want_idx = ref.optimize_chunk_size(trace, sizes)
    try:
        got_idx = optimize_chunk_size(trace, sizes)
        if got_idx == want_idx:
            out["argmin_index"] = 1.0
        else:
            out["_note_opt"] = f"got idx {got_idx}, want {want_idx}"
    except Exception:
        pass

    return out
