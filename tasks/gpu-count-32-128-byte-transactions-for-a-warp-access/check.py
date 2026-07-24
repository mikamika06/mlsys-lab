import math

def _ref_count_transactions(base_addr: int, stride: int, num_threads: int) -> int:
    """Brute-force reference: enumerate all addresses and count distinct segments."""
    segments = set()
    for i in range(num_threads):
        addr = base_addr + i * stride
        segments.add(addr // 128)
    return len(segments)

def grade(sol, fx) -> dict:
    """Grade count_transactions against the brute-force reference."""
    test_cases = [
        (0, 4, 32, "coalesced float32"),
        (0, 8, 32, "stride-2 float32"),
        (64, 4, 32, "coalesced float32 offset by 64 bytes"),
        (0, 128, 32, "fully scattered"),
        (1, 4, 32, "coalesced, unaligned base"),
        (1, 5, 32, "odd stride, unaligned base"),
        (0, 16, 32, "stride-4 float64"),
        (256, 4, 32, "coalesced, base at segment boundary"),
        (100, 7, 32, "arbitrary stride and base"),
        (0, 4, 1, "single thread"),
        (0, 4, 16, "half warp coalesced"),
        (0, 256, 32, "each thread in separate segment"),
        (300, 12, 32, "misaligned, strided"),
        (127, 1, 32, "all in one segment, unaligned"),
        (0, 3, 32, "stride-3 access pattern"),
    ]

    all_pass = True
    for base, stride, nthreads, _desc in test_cases:
        try:
            got = sol.count_transactions(base, stride, nthreads)
        except Exception:
            all_pass = False
            continue
        expected = _ref_count_transactions(base, stride, nthreads)
        if int(got) != expected:
            all_pass = False

    return {"all_correct": all_pass}
