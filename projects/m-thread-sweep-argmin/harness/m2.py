import ref
import numpy as np

def check(workdir):
    from ortopt.iobind import run_with_iobinding
    from ortopt.metrics import compute_copy_share

    out = {"iobinding_outputs_match": 0.0, "copy_share_valid": 0.0}
    try:
        inputs = [np.array([10.0, 20.0], dtype=np.float32)]
        got_outs = run_with_iobinding(None, inputs)
        want_outs = ref.run_with_iobinding(None, inputs)

        match = len(got_outs) == len(want_outs)
        if match:
            for g, w in zip(got_outs, want_outs):
                if not np.allclose(g, w):
                    match = False
                    break
        if match:
            out["iobinding_outputs_match"] = 1.0
        else:
            out["_note"] = "run_with_iobinding outputs did not match reference"

        share = compute_copy_share(200.0, 50.0)
        want_share = ref.compute_copy_share(200.0, 50.0)
        if np.isclose(share, want_share):
            out["copy_share_valid"] = 1.0
        else:
            out["_note"] = f"compute_copy_share got {share}, want {want_share}"
    except Exception as e:
        out["_note"] = f"exception in milestone 2: {type(e).__name__}: {str(e)[:120]}"
    return out
