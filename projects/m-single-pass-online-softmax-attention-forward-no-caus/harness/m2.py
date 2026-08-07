import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from fused_attn.causal import block_split_causal_attention_forward

    out = {"rel_err": 1.0, "causal_tiles_matched": 0.0}
    Q, K, V, sm_scale = ref.generate_inputs(B=2, H=2, N=64, D=16, seed=123)
    expected = ref.standard_attention_forward(Q, K, V, sm_scale, causal=True)

    try:
        got = block_split_causal_attention_forward(Q, K, V, sm_scale, block_size=16)
        err = np.max(np.abs(got - expected) / (np.abs(expected) + 1e-8))
        out["rel_err"] = float(err)

        upper_tri_valid = True
        for i in range(Q.shape[2]):
            for j in range(i + 1, Q.shape[2]):
                V_mod = V.copy()
                V_mod[:, :, j, :] += 100.0
                got_mod = block_split_causal_attention_forward(Q, K, V_mod, sm_scale, block_size=16)
                if not np.allclose(got[:, :, i, :], got_mod[:, :, i, :]):
                    upper_tri_valid = False
                    break
            if not upper_tri_valid:
                break

        if upper_tri_valid:
            out["causal_tiles_matched"] = 1.0
        else:
            out["_note"] = "Causal masking failed: upper triangular key/value tokens affected earlier queries."
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {str(e)}"

    return out
