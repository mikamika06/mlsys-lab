import ref
import numpy as np


def check(workdir):
    from swm.masking import generate_sliding_window_mask
    from swm.attention import windowed_attention

    out = {"mask_correct": 0.0, "attn_correct": 0.0}

    w_want = ref.generate_sliding_window_mask(100, 15)
    w_got = generate_sliding_window_mask(100, 15)
    if np.array_equal(w_want, w_got):
        out["mask_correct"] = 1.0
    else:
        out["_note"] = "Sliding window mask does not match reference."

    q, k, v = ref.get_m1_fixtures()
    mask = ref.generate_sliding_window_mask(16, 4)
    a_want = ref.windowed_attention(q, k, v, mask)
    a_got = windowed_attention(q, k, v, mask)

    if np.allclose(a_want, a_got, atol=1e-5):
        out["attn_correct"] = 1.0
    elif "_note" not in out:
        out["_note"] = "Windowed attention outputs do not match reference."

    return out
