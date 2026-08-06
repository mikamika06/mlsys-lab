import numpy as np
import ref


def check(workdir):
    from awqsim.scale import compute_awq_scales, derive_awq_mappings

    fixtures = ref.generate_fixtures()
    out = {"mappings_matched": 0.0, "scales_matched": 0.0}

    map_ok = 0
    for cfg in fixtures["configs"]:
        want = ref.derive_awq_mappings(cfg)
        got = derive_awq_mappings(cfg)
        if got == want:
            map_ok += 1
        elif "_note" not in out:
            out["_note"] = f"mapping mismatch: got {got}, want {want}"

    out["mappings_matched"] = float(map_ok)

    scale_ok = 0
    scale_tests = [
        ([fixtures["W1"]], fixtures["X1"], 0.5, 5.0),
        ([fixtures["W2_a"], fixtures["W2_b"]], fixtures["X2"], 0.5, 5.0),
        ([fixtures["W2_a"]], fixtures["X2"], 0.25, 3.0),
    ]

    for W_list, X, alpha, ratio in scale_tests:
        want_scale = ref.compute_awq_scales(
            W_list, X, alpha=alpha, max_scale_ratio=ratio
        )
        got_scale = compute_awq_scales(
            W_list, X, alpha=alpha, max_scale_ratio=ratio
        )
        if got_scale is not None and isinstance(got_scale, np.ndarray):
            if np.allclose(got_scale, want_scale, atol=1e-4):
                scale_ok += 1
            elif "_note" not in out:
                out["_note"] = (
                    f"scale mismatch: max diff {np.max(np.abs(got_scale - want_scale))}"
                )
        elif "_note" not in out:
            out["_note"] = "compute_awq_scales did not return numpy array"

    out["scales_matched"] = float(scale_ok)
    return out
