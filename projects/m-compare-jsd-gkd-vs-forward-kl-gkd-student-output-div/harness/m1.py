import ref
import numpy as np

def check(workdir):
    from gkd.diversity import compute_token_entropy, compute_vocabulary_coverage
    logits_jsd, logits_fkl, tokens_jsd, tokens_fkl = ref.generate_fixtures()

    want_ent = ref.compute_entropy(logits_jsd)
    try:
        got_ent = compute_token_entropy(logits_jsd)
    except Exception as e:
        return {"entropy_matched": 0.0, "_note": f"exception: {e}"}

    want_cov = len(np.unique(tokens_jsd)) / 50.0
    try:
        got_cov = compute_vocabulary_coverage(tokens_jsd, 50)
    except Exception as e:
        return {"entropy_matched": 0.0, "_note": f"exception: {e}"}

    ok = 1.0 if (np.isclose(got_ent, want_ent, atol=1e-4) and np.isclose(got_cov, want_cov, atol=1e-4)) else 0.0
    return {"entropy_matched": ok}
