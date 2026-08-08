import ref


def check(workdir):
    from core_image.export import simulate_conversion
    from core_image.metrics import compute_drift_and_ratio

    out = {"drift_verified": 0.0, "ratio_matched": 0.0}
    weights = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    fp32_res = simulate_conversion(weights, "fp32")
    fp16_res = simulate_conversion(weights, "fp16")

    want_drift, want_ratio = ref.compute_drift_and_ratio(
        fp32_res["weights"], fp16_res["weights"], fp32_res["size_bytes"], fp16_res["size_bytes"]
    )

    try:
        got_drift, got_ratio = compute_drift_and_ratio(
            fp32_res["weights"], fp16_res["weights"], fp32_res["size_bytes"], fp16_res["size_bytes"]
        )
    except Exception:
        got_drift, got_ratio = -1.0, -1.0

    if abs(got_drift - want_drift) < 1e-5:
        out["drift_verified"] = 1.0
    if abs(got_ratio - want_ratio) < 1e-5:
        out["ratio_matched"] = 1.0

    if out["drift_verified"] == 0.0:
        out["_note"] = f"drift mismatch: got {got_drift}, want {want_drift}"
    elif out["ratio_matched"] == 0.0:
        out["_note"] = f"ratio mismatch: got {got_ratio}, want {want_ratio}"

    return out
