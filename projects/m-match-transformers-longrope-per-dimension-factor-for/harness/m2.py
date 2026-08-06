import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from longrope.perplexity import compute_rope_inv_freqs, evaluate_synthetic_perplexity
    from longrope.entropy import measure_yarn_attention_entropy

    out = {"ppl_ranks_matched": 0.0, "entropy_trends_matched": 0.0}

    np.random.seed(42)
    seq_len = 1024
    vocab_size = 500
    head_dim = 64
    orig_len = 4096
    target_len = 16384
    scale = target_len / orig_len

    logits = np.random.randn(seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, size=(seq_len,))

    methods = ["linear", "dynamic_ntk", "yarn"]
    ppls = {}
    for m in methods:
        inv_freqs = compute_rope_inv_freqs(m, head_dim, orig_len, target_len, scale_factor=scale)
        ppl = evaluate_synthetic_perplexity(logits, targets, inv_freqs, seq_len, scale_factor=scale)
        ppls[m] = ppl

    ref_ppls = {}
    for m in methods:
        inv_freqs = ref.oracle_inv_freqs(m, head_dim, orig_len, target_len, scale_factor=scale)
        ppl = ref.oracle_perplexity(logits, targets, inv_freqs, seq_len, scale_factor=scale)
        ref_ppls[m] = ppl

    learner_order = sorted(ppls, key=ppls.get)
    ref_order = sorted(ref_ppls, key=ref_ppls.get)

    if learner_order == ref_order and np.allclose([ppls[m] for m in methods], [ref_ppls[m] for m in methods], rtol=1e-3):
        out["ppl_ranks_matched"] = 1.0
    else:
        out["_note"] = f"PPL mismatch. Learner: {ppls}, Reference: {ref_ppls}"

    q = np.random.randn(seq_len, head_dim)
    k = np.random.randn(seq_len, head_dim)
    yarn_freqs = compute_rope_inv_freqs("yarn", head_dim, orig_len, target_len, scale_factor=scale)

    ent_low = measure_yarn_attention_entropy(q, k, yarn_freqs, attention_factor=0.5)
    ent_mid = measure_yarn_attention_entropy(q, k, yarn_freqs, attention_factor=1.0)
    ent_high = measure_yarn_attention_entropy(q, k, yarn_freqs, attention_factor=2.0)

    if ent_low > ent_mid > ent_high:
        out["entropy_trends_matched"] = 1.0
    else:
        out["_note"] = f"Entropy trend expected high->low as attn_factor increases. Got: 0.5->{ent_low:.3f}, 1.0->{ent_mid:.3f}, 2.0->{ent_high:.3f}"

    return out
