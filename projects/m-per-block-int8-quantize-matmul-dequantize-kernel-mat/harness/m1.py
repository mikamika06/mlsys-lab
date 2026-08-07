import ref
import sys

def check(workdir):
    out = {"matmul_rel_err": 1.0}
    try:
        import qmat.matmul as matmul
        A, B = ref.generate_matmul_fixtures()
        want = ref.per_block_int8_matmul(A, B, 16)
        got = matmul.per_block_int8_matmul(A, B, 16)
        err = ref.np.max(ref.np.abs(want - got)) / (ref.np.max(ref.np.abs(want)) + 1e-9)
        out["matmul_rel_err"] = float(err)
    except Exception as e:
        out["_note"] = f"m1 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
