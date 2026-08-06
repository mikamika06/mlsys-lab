import ref


def check(workdir):
    from tritoncache.cache_key import build_cache_key

    out = {"keys_matched": 0.0}
    cases = ref.get_test_cases()
    matched = 0
    total = 0

    for case in cases:
        fn_name = case["fn_name"]
        sig = case["sig"]
        for run in case["runs"]:
            total += 1
            got = build_cache_key(fn_name, sig, run)

            parts = [f"fn:{fn_name}"]
            for param_name, meta in sig.items():
                is_constexpr = meta.get("is_constexpr", False)
                is_ptr = meta.get("is_ptr", False)
                val = run[param_name]
                if is_constexpr:
                    c = ("constexpr", param_name, repr(val))
                elif is_ptr:
                    c = (
                        "tensor",
                        param_name,
                        str(getattr(val, "dtype", "ptr")),
                        tuple(getattr(val, "shape", ())),
                        tuple(getattr(val, "stride", ())),
                    )
                else:
                    c = ("scalar", param_name, type(val).__name__)
                parts.append(f"{param_name}:{c}")
            want = "|".join(parts)

            if got == want:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Key mismatch. Got: {got} | Want: {want}"

    if matched == total and total > 0:
        out["keys_matched"] = 1.0
    return out
