import ref


def check(workdir):
    from seqcomm.formulas import (
        ulysses_comm_volume_per_layer,
        ring_comm_volume_per_layer,
        usp_comm_volume_per_layer,
    )

    cases = ref.generate_analytical_cases()
    matched = 0
    total = len(cases)

    for c in cases:
        N, D, P, b = c["seq_len"], c["hidden_dim"], c["world_size"], c["dtype_bytes"]

        want_u = 2 * (2 * (P - 1) * (N // P) * D * b + 2 * (P - 1) * (N // P) * D * b)
        got_u = ulysses_comm_volume_per_layer(N, D, P, b)

        want_r = 4 * (P - 1) * 2 * (N // P) * D * b
        got_r = ring_comm_volume_per_layer(N, D, P, b)

        u_deg, r_deg = 2, P // 2
        want_usp = ulysses_comm_volume_per_layer(N, D, u_deg, b) + ring_comm_volume_per_layer(N // u_deg, D, r_deg, b)
        got_usp = usp_comm_volume_per_layer(N, D, P, u_deg, r_deg, b)

        if got_u == want_u and got_r == want_r and got_usp == want_usp:
            matched += 1

    return {
        "formulas_matched": 1.0 if matched == total else 0.0,
        "_note": f"Matched {matched}/{total} formula cases"
    }
