import numpy as np
import ref


def check(workdir):
    from distill.mapping import build_tinybert_layer_mapping
    from distill.attention import compute_attention_kl_divergence

    out = {"mapping_matches": 0.0, "rel_err": 1.0}

    ref_map = ref.ref_build_tinybert_layer_mapping(4, 12)
    try:
        user_map = build_tinybert_layer_mapping(4, 12)
        if user_map == ref_map:
            out["mapping_matches"] = 1.0
        else:
            out["_note"] = f"Expected mapping {ref_map}, got {user_map}"
    except Exception as e:
        out["_note"] = f"Mapping error: {e}"
        return out

    fixtures = ref.generate_fixtures(seed=202)
    s_attn = fixtures["student_attn"]
    t_attn = fixtures["teacher_attn"]

    ref_kl = ref.ref_compute_attention_kl_divergence(s_attn, t_attn)
    try:
        user_kl = compute_attention_kl_divergence(s_attn, t_attn)
        err = abs(user_kl - ref_kl) / (abs(ref_kl) + 1e-12)
        out["rel_err"] = float(err)
    except Exception as e:
        out["_note"] = f"KL calculation error: {e}"

    return out
