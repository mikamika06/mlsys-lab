def grade(sol, fx) -> dict:
    def ref(n, d, B, line_size):
        L = line_size // 8
        rows_per_line = (d + L - 1) // L
        naive = n * n * 2 * rows_per_line

        # compute block sizes
        blocks = []
        full_blocks = n // B
        rem = n % B
        for _ in range(full_blocks):
            blocks.append(B)
        if rem:
            blocks.append(rem)

        tiled = 0
        for sz_i in blocks:
            for sz_j in blocks:
                tiled += 2 * rows_per_line * (sz_i + sz_j)
        return naive, tiled

    cases = [
        (5, 3, 2, 64),
        (10, 8, 4, 32),
        (7, 15, 3, 128),
        (12, 20, 5, 64),
        (1, 1, 1, 64)
    ]

    ok = 1.0
    for n, d, B, line_size in cases:
        try:
            got = sol.pairwise_memory_access(n, d, B, line_size)
            if not isinstance(got, tuple) or len(got) != 2:
                ok = 0.0
                break
            ref_vals = ref(n, d, B, line_size)
            if got != ref_vals:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
