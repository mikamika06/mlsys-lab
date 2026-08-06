from matrix.canary import validate_canary_artifact


def test_canary_layout_regression():
    ref_trace = {
        "shape": [1, 128, 768],
        "dtype": "float32",
        "layout": "NCHW",
        "data": [0.1] * (1 * 128 * 768),
    }

    good_cand = {
        "shape": [1, 128, 768],
        "dtype": "float32",
        "layout": "NCHW",
        "data": [0.10001] * (1 * 128 * 768),
    }

    bad_layout_cand = {
        "shape": [1, 128, 768],
        "dtype": "float32",
        "layout": "NHWC",
        "data": [0.1] * (1 * 128 * 768),
    }

    assert validate_canary_artifact(good_cand, ref_trace) is True
    assert validate_canary_artifact(bad_layout_cand, ref_trace) is False
