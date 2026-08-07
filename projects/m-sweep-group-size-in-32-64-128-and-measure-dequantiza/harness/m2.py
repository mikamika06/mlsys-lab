import numpy as np
import ref


def check(workdir):
    out = {"hand_dequant_matched": 0.0, "bit_comparison_matched": 0.0}
    try:
        from mlx_quant.compare import compare_bit_widths
        from mlx_quant.unpack import pack_uint4_pair, unpack_and_dequantize_4bit
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {e}"
        return out

    weights = ref.generate_test_weights(shape=(128, 128), seed=999)
    group_size = 32

    try:
        qw, scales, biases = ref.quantize_affine(weights, group_size=group_size, bits=4)
        packed = ref.pack_uint4_pair(qw)

        got_deq = unpack_and_dequantize_4bit(
            packed, scales, biases, group_size, original_shape=weights.shape
        )
        want_deq = ref.unpack_and_dequantize_4bit(
            packed, scales, biases, group_size, original_shape=weights.shape
        )

        if np.allclose(got_deq, want_deq, rtol=1e-4, atol=1e-5):
            out["hand_dequant_matched"] = 1.0
        else:
            out["_note"] = "Hand dequantization output does not match expected output"
    except Exception as e:
        out["_note"] = f"Unpack error: {type(e).__name__}: {e}"
        return out

    try:
        got_cmp = compare_bit_widths(weights, group_size=64)
        want_cmp = ref.compare_bit_widths(weights, group_size=64)

        b4_ok = got_cmp.get("4bit", {}).get("bytes") == want_cmp["4bit"]["bytes"]
        b8_ok = got_cmp.get("8bit", {}).get("bytes") == want_cmp["8bit"]["bytes"]
        mse4_ok = np.isclose(
            got_cmp.get("4bit", {}).get("mse", -1), want_cmp["4bit"]["mse"], rtol=1e-4
        )
        mse8_ok = np.isclose(
            got_cmp.get("8bit", {}).get("mse", -1), want_cmp["8bit"]["mse"], rtol=1e-4
        )

        if b4_ok and b8_ok and mse4_ok and mse8_ok:
            out["bit_comparison_matched"] = 1.0
        else:
            out["_note"] = f"Bit comparison mismatch. Got: {got_cmp}, Want: {want_cmp}"
    except Exception as e:
        out["_note"] = f"Compare error: {type(e).__name__}: {e}"

    return out
