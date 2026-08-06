import ref

def check(workdir):
    import numpy as np
    from dfkd.sampling import get_transition_probs, sample_teacher

    out = {"probs_match": 0.0, "sequence_match": 0.0}
    t_logits, _ = ref.generate_fixtures()

    try:
        p_want = ref.get_transition_probs(t_logits)
        p_got = get_transition_probs(t_logits)
        if np.allclose(p_want, p_got):
            out["probs_match"] = 1.0

        seq_want = ref.sample_teacher(t_logits, 0, 50, 42)
        seq_got = sample_teacher(t_logits, 0, 50, 42)
        if np.array_equal(seq_want, seq_got):
            out["sequence_match"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
