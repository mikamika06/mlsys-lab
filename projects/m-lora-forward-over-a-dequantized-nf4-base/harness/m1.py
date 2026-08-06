import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from qlora.forward import lora_nf4_forward, dequantize_nf4
    except ImportError as e:
        return {"forward_matched": 0, "_note": f"Import error: {e}"}

    np.random.seed(42)
    matched = 0

    for i in range(3):
        block_size = 32
        in_features, out_features = 32, 16
        r = 4
        num_blocks = (in_features * out_features) // block_size

        qweight = np.random.randint(0, 16, size=(num_blocks, block_size), dtype=np.int32)
        absmax = np.random.uniform(0.1, 2.0, size=(num_blocks,)).astype(np.float32)
        lora_a = np.random.randn(r, in_features).astype(np.float32)
        lora_b = np.random.randn(out_features, r).astype(np.float32)
        x = np.random.randn(4, in_features).astype(np.float32)
        scaling = 2.0

        want = ref.lora_nf4_forward(x, qweight, absmax, ref.NF4_CODEBOOK, lora_a, lora_b, scaling, "float32", block_size)
        try:
            got = lora_nf4_forward(x, qweight, absmax, ref.NF4_CODEBOOK, lora_a, lora_b, scaling, "float32", block_size)
            if np.allclose(want, got, atol=1e-4):
                matched += 1
            else:
                return {"forward_matched": float(matched), "_note": f"Mismatch in case {i}: max abs diff = {np.max(np.abs(want - got))}"}
        except Exception as e:
            return {"forward_matched": float(matched), "_note": f"Exception in case {i}: {e}"}

    return {"forward_matched": float(matched)}
