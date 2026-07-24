import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    shapes = [(10,), (50,), (200,)]
    max_fp16_err = 0.0
    max_bf16_err = 0.0

    for shape in shapes:
        arr = rng.uniform(-1e3, 1e3, size=shape).astype(np.float32)

        # Reference FP16: cast to float16 then back to float32
        fp16_ref = arr.astype(np.float16).astype(np.float32)

        # Reference BF16: truncate lower 16 bits of the uint32 representation
        bits = arr.view(np.uint32)
        bf16_bits = (bits >> 16) & 0xFFFF
        bf16_arr = (bf16_bits.astype(np.uint32) << 16).view(np.float32)

        try:
            fp16_out, bf16_out = sol.compare_rounding(arr)
        except Exception:
            # If the solution crashes, set errors to a large value
            return {"max_fp16_err": float("inf"), "max_bf16_err": float("inf")}

        # Compute maximum absolute error for each format
        fp16_err = np.max(np.abs(fp16_out.astype(np.float32) - fp16_ref))
        bf16_err = np.max(np.abs(bf16_out.astype(np.float32) - bf16_arr))

        max_fp16_err = max(max_fp16_err, fp16_err)
        max_bf16_err = max(max_bf16_err, bf16_err)

    return {"max_fp16_err": float(max_fp16_err), "max_bf16_err": float(max_bf16_err)}
