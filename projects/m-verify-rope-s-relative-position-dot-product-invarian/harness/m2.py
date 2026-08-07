import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from rope.core import compute_rope_frequencies, apply_position_interpolation, compute_perplexity

    out = {"max_abs_err": 0.0, "unscaled_ppl_ratio": 0.0}

    dim = 32
    vocab_size = 50
    trained_len = 254
    extended_len = 1016
    scale_factor = 4.0

    q, k, logits, targets = ref.generate_synthetic_data(dim=dim, seq_len=extended_len, vocab_size=vocab_size, seed=777)

    user_ppl = compute_perplexity(logits, targets)
    ref_ppl = ref.compute_perplexity(logits, targets)

    ppl_err = float(np.abs(user_ppl - ref_ppl))
    if ppl_err > 1e-5:
        out["max_abs_err"] = ppl_err
        out["_note"] = f"Perplexity calculation error: got {user_ppl}, want {ref_ppl}"
        return out

    pos_out = apply_position_interpolation(np.arange(extended_len), scale_factor)
    ref_pos = ref.apply_position_interpolation(np.arange(extended_len), scale_factor)
    pos_err = float(np.max(np.abs(pos_out - ref_pos)))

    out["max_abs_err"] = max(ppl_err, pos_err)

    rng = np.random.RandomState(42)
    bad_logits = logits.copy()
    bad_logits[trained_len:] += rng.randn(extended_len - trained_len, vocab_size) * 10.0

    unscaled_ppl = ref.compute_perplexity(bad_logits, targets)
    normal_ppl = ref.compute_perplexity(logits[:trained_len], targets[:trained_len])

    ratio = float(unscaled_ppl / normal_ppl)
    out["unscaled_ppl_ratio"] = ratio

    return out
