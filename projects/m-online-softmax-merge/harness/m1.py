import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"blocks_matched": 0.0}

    try:
        from onlinesoftmax.merge import merge_online_softmax, chunked_online_attention
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    rng = np.random.default_rng(42)
    m_a = rng.normal(size=(8,))
    l_a = rng.uniform(0.5, 2.0, size=(8,))
    o_a = rng.normal(size=(8, 16))

    m_b = rng.normal(size=(8,))
    l_b = rng.uniform(0.5, 2.0, size=(8,))
    o_b = rng.normal(size=(8, 16))

    ref_m, ref_l, ref_o = ref.reference_merge_online_softmax(m_a, l_a, o_a, m_b, l_b, o_b)

    try:
        got_m, got_l, got_o = merge_online_softmax(m_a, l_a, o_a, m_b, l_b, o_b)
    except Exception as e:
        out["_note"] = f"merge_online_softmax raised exception: {e}"
        return out

    if not (np.allclose(got_m, ref_m) and np.allclose(got_l, ref_l) and np.allclose(got_o, ref_o)):
        out["_note"] = "merge_online_softmax output mismatch"
        return out

    q = rng.normal(size=(4, 16))
    k = rng.normal(size=(128, 16))
    v = rng.normal(size=(128, 16))

    ref_attn = ref.reference_chunked_attention(q, k, v, chunk_size=32)
    try:
        got_attn = chunked_online_attention(q, k, v, chunk_size=32)
    except Exception as e:
        out["_note"] = f"chunked_online_attention raised exception: {e}"
        return out

    if not np.allclose(got_attn, ref_attn, rtol=1e-8, atol=1e-8):
        out["_note"] = "chunked_online_attention output mismatch against reference"
        return out

    out["blocks_matched"] = 1.0
    return out
