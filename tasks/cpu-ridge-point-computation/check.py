from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    """Grade ridge-point computation and cache-friendly dot-product trace."""
    n, l1_bytes, line_bytes = 2048, 16384, 64
    peak_flops, peak_bw = 4e12, 200e9  # 4 TFLOP/s, 200 GB/s

    # --- ridge-point gate ---
    try:
        ans = float(sol.ridge_point(peak_flops, peak_bw))
        ref = peak_flops / peak_bw
        rel_err = abs(ans - ref) / abs(ref) if ref != 0 else (1.0 if ans != 0 else 0.0)
    except Exception:
        rel_err = 1.0

    # --- dot-trace gates ---
    covers = 0.0
    misses = 10**9
    try:
        trace = list(sol.dot_trace(n, l1_bytes, line_bytes))
        # Check every element (not every byte) of both arrays is visited.
        # Array a has n float64 elements at byte addrs 0..8n-1,
        # array b has n float64 elements at byte addrs 8n..16n-1.
        # Element index = byte_addr // 8; total unique elements = 2*n.
        elem_indices = set(addr // 8 for addr in trace)
        expected_elems = set(range(2 * n))
        covers = 1.0 if elem_indices == expected_elems else 0.0

        addrs = [int(a) for a in trace]
        misses = cachesim.simulate(
            addrs, line_bytes=line_bytes, sets=64, ways=8
        )["misses"]
    except Exception:
        pass

    return {
        "rel_err": rel_err,
        "covers_all": covers,
        "misses": misses,
    }
