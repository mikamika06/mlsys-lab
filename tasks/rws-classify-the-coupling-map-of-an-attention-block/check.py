def _reference(q_shape, k_shape, v_shape, o_proj_shape, head_index):
    # All shapes are tuples; we only need the last dimension for Q/K/V and first for O_proj.
    d_k = q_shape[-1] // (head_index + 1) * (head_index + 1) - \
          q_shape[-1] // (head_index + 1) * head_index
    # The above computes block size per head assuming equal partitioning.
    # A simpler robust way:
    total_heads_q = q_shape[-1] // d_k if 'd_k' in locals() else None

def _ref(q_shape, k_shape, v_shape, o_proj_shape, head_index):
    # Determine number of heads and head dimension from shapes
    Hq = q_shape[-1]
    Hv = o_proj_shape[0]
    # Assume equal partitioning: d_k = Hq // H, d_v = Hv // H
    # We infer H by gcd of dimensions? For simplicity assume same H for all.
    # Compute H as the greatest common divisor of Hq and Hv
    import math
    H = math.gcd(Hq, Hv)
    if H == 0:
        raise ValueError("Invalid shapes")
    d_k = Hq // H
    d_v = Hv // H
    start_qk = head_index * d_k
    end_qk = (head_index + 1) * d_k
    start_o = head_index * d_v
    end_o = (head_index + 1) * d_v
    return {
        "q": (start_qk, end_qk),
        "k": (start_qk, end_qk),
        "v": (start_qk, end_qk),
        "o_proj_input": (start_o, end_o)
    }

def grade(sol, fx) -> dict:
    # Define a set of deterministic test cases
    cases = [
        ((2, 10, 64), (2, 10, 64), (2, 10, 64), (64, 128), 0),
        ((2, 10, 64), (2, 10, 64), (2, 10, 64), (64, 128), 1),
        ((2, 10, 64), (2, 10, 64), (2, 10, 64), (64, 128), 3),
        ((4, 20, 80), (4, 20, 80), (4, 20, 80), (80, 256), 2),
        ((1, 5, 48), (1, 5, 48), (1, 5, 48), (48, 64), 4)
    ]
    ok = 1.0
    for q_shape, k_shape, v_shape, o_proj_shape, head_index in cases:
        try:
            got = sol.classify_coupling_map(q_shape, k_shape, v_shape, o_proj_shape, head_index)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(q_shape, k_shape, v_shape, o_proj_shape, head_index)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
