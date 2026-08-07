import ref

def check(workdir):
    from det.sampling import check_determinism, break_determinism_state

    seed = 123
    b1 = ref.generate_bytes(seed, 0.0, [1, 2, 3])
    b2 = ref.generate_bytes(seed, 0.0, [1, 2, 3])

    try:
        user_b1 = check_determinism(seed, 0.0, [1, 2, 3])
        user_b2 = check_determinism(seed, 0.0, [1, 2, 3])
        match_ok = 1.0 if user_b1 == user_b2 == b1 else 0.0
    except Exception:
        match_ok = 0.0

    try:
        broken = break_determinism_state(seed, [1, 2, 3])
        break_ok = 1.0 if broken else 0.0
    except Exception:
        break_ok = 0.0

    return {
        "byte_exact_fraction": match_ok,
        "broken_fraction": break_ok
    }
