import ref
import numpy as np

def check(workdir):
    from gkd.metrics import compare_gkd_diversity
    logits_jsd, logits_fkl, tokens_jsd, tokens_fkl = ref.generate_fixtures()
    want_res = ref.compute_diversity_ratio(logits_jsd, logits_fkl, tokens_jsd, tokens_fkl)

    try:
        got_dict = compare_gkd_diversity(logits_jsd, logits_fkl, tokens_jsd, tokens_fkl, vocab_size=50)
        got_ratio = got_dict["ratio"]
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"exception: {e}"}

    rel = abs(got_ratio - want_res) / (abs(want_res) + 1e-12)
    return {"rel_err": float(rel)}
