import ref


def check(workdir):
    from quant.blocks import find_optimal_distribution_params

    out = {"distribution_matched": 0.0}
    try:
        got = find_optimal_distribution_params()
        want = ref.get_expected_distribution()
        if isinstance(got, dict) and "skew" in got and abs(got["skew"] - want["skew"]) < 1e-5:
            out["distribution_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, expected around skew {want['skew']}"
    except Exception as e:
        out["_note"] = f"error executing find_optimal_distribution_params: {type(e).__name__}: {e}"
    return out
