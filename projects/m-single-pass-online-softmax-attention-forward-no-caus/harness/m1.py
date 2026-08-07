import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from fused_attn.online_softmax import online_softmax_attention_forward

    out = {"rel_err": 1.0}
    Q, K, V, sm_scale = ref.generate_inputs(B=2, H=2, N=64, D=16, seed=42)
    expected = ref.standard_attention_forward(Q, K, V, sm_scale, causal=False)

    try:
        got = online_softmax_attention_forward(Q, K, V, sm_scale)
        err = np.max(np.abs(got - expected) / (np.abs(expected) + 1e-8))
        out["rel_err"] = float(err)
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {str(e)}"

    return out
