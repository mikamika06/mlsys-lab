import ref
import numpy as np


def check(workdir):
    from int8opt.scales import recover_per_channel_scales

    out = {"scales_matched": 0.0}
    try:
        mock = ref.generate_mock_onnx()
        got = recover_per_channel_scales(mock)
        want = ref.recover_per_channel_scales(mock)
        if got is not None and np.allclose(got, want):
            out["scales_matched"] = 1.0
        else:
            out["_note"] = f"got scales {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error during execution: {str(e)[:120]}"
    return out
