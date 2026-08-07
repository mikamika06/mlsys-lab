import ref


def check(workdir):
    out = {"native_replaced": 0.0}
    try:
        from edge_mlx import ops
        x = ref.MockTensor([1.0, 2.0, 3.0], device="mps")
        res = ops.native_unsupported_op(x)
        if res is not None and res.device == "mps":
            out["native_replaced"] = 1.0
        else:
            out["_note"] = "native_unsupported_op failed or returned incorrect device"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:100]}"
    return out
