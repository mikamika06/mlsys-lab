import ref


def check(workdir):
    out = {"fft_fixed": 0.0}
    try:
        from edge_mlx import fft
        x = ref.MockTensor([1.0, 2.0, 3.0, 4.0], device="mps")
        res = fft.safe_fft_r2c(x)
        if res is not None and res.device == "mps":
            out["fft_fixed"] = 1.0
        else:
            out["_note"] = "safe_fft_r2c did not return a valid mps tensor"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:100]}"
    return out
