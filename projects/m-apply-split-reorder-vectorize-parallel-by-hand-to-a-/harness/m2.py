import ref


def check(workdir):
    out = {"speedup_ratio": 0.0}
    try:
        from tirsched.schedule import create_naive_matmul, apply_split_reorder_vectorize_parallel, measure_speedup
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    m, n, k, a_np, b_np, _ = ref.generate_fixtures(seed=2025)
    naive_mod = create_naive_matmul(m, n, k)
    steps = apply_split_reorder_vectorize_parallel(naive_mod, factors=(16, 16))
    scheduled_mod = steps[-1][1]

    try:
        ratio = measure_speedup(naive_mod, scheduled_mod, a_np, b_np)
        out["speedup_ratio"] = float(ratio)
    except Exception as e:
        out["_note"] = f"measure_speedup raised exception: {e}"

    return out
