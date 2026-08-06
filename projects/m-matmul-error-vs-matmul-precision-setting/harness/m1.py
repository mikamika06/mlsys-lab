import ref
import numpy as np

def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"tf32_truncation": 0.0, "bf16_truncation": 0.0, "chain_correct": 0.0}
    try:
        import numerics
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    x = np.random.RandomState(42).randn(100).astype(np.float32)
    try:
        if np.array_equal(numerics.truncate_to_tf32(x), ref.truncate_to_tf32(x)):
            out["tf32_truncation"] = 1.0
    except Exception:
        pass

    try:
        if np.array_equal(numerics.truncate_to_bf16(x), ref.truncate_to_bf16(x)):
            out["bf16_truncation"] = 1.0
    except Exception:
        pass

    try:
        ok = 0
        for prec in ["fp32", "tf32", "bf16"]:
            got = numerics.matmul_chain(ref.MATRIX_CHAIN, prec)
            want = ref.matmul_chain(ref.MATRIX_CHAIN, prec)
            if np.allclose(got, want, atol=1e-5, rtol=1e-5):
                ok += 1
        if ok == 3:
            out["chain_correct"] = 1.0
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"matmul_chain failed: {e}"

    return out
