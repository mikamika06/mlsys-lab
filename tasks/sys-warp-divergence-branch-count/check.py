def grade(sol, fx) -> dict:
    fn = getattr(sol, "warp_divergence_branch_count", None)
    if fn is None:
        return {"exact_match": 0.0}

    def ref_warp_divergence_branch_count(preds, warp_size=32):
        if not isinstance(preds, list) or any(isinstance(item, list) for item in preds):
            raise ValueError("preds must be a 1D list")
        n = len(preds)
        if n % warp_size != 0:
            raise ValueError(f"Length {n} is not a multiple of warp_size {warp_size}")
        num_blocks = n // warp_size
        out = []
        for i in range(num_blocks):
            block = preds[i * warp_size : (i + 1) * warp_size]
            seen = []
            for item in block:
                if item not in seen:
                    seen.append(item)
            out.append(len(seen))
        return out

    test_cases = [
        ([0, 1] * 16, 32),
        ([0] * 64 + [1] * 64, 32),
        ([0, 1, 2, 3] * 8, 16),
        ([i % 7 for i in range(128)], 32),
        ([1] * 96, 32),
    ]

    for preds, warp_size in test_cases:
        try:
            expected = ref_warp_divergence_branch_count(preds, warp_size)
            got = fn(preds, warp_size)
            if got != expected:
                return {"exact_match": 0.0}
        except Exception:
            return {"exact_match": 0.0}

    # Test default parameter (warp_size=32)
    try:
        preds_default = [0, 1] * 16
        if fn(preds_default) != ref_warp_divergence_branch_count(preds_default):
            return {"exact_match": 0.0}
    except Exception:
        return {"exact_match": 0.0}

    # Verify exception when input length is not a multiple of warp_size
    try:
        fn([0] * 30, 32)
        return {"exact_match": 0.0}
    except ValueError:
        pass
    except Exception:
        return {"exact_match": 0.0}

    # Verify exception when input is not a 1D list
    try:
        fn([[0, 1], [1, 0]], 2)
        return {"exact_match": 0.0}
    except ValueError:
        pass
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
