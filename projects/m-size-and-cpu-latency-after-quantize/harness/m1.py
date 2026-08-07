import ref


def check(workdir):
    from quant.bench import profile_quantization

    out = {"size_ratio": 0.0, "speedup_match": 0.0}

    ref_profile = ref.profile_quantization(
        ref.BEFORE_MODEL_1, ref.AFTER_MODEL_1, ref.BENCH_FN_1
    )
    user_profile = profile_quantization(
        ref.BEFORE_MODEL_1, ref.AFTER_MODEL_1, ref.BENCH_FN_1
    )

    ref_ratio = ref_profile["size_ratio"]
    user_ratio = user_profile.get("size_ratio", 0.0)

    if abs(ref_ratio - user_ratio) < 1e-5:
        out["size_ratio"] = 1.0
    else:
        out["_note"] = f"size_ratio mismatch: expected {ref_ratio}, got {user_ratio}"

    ref_speedup = ref_profile["speedup"]
    user_speedup = user_profile.get("speedup", 0.0)

    if abs(ref_speedup - user_speedup) < 1e-5:
        out["speedup_match"] = 1.0
    elif "_note" not in out:
        out["_note"] = (
            f"speedup mismatch: expected {ref_speedup}, got {user_speedup}"
        )

    return out
