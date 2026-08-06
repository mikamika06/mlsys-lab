import ref

def check(workdir):
    import numpy as np
    from dfkd.distill import fit_student_low_rank, compare_distillation

    out = {"student_fit_match": 0.0, "accuracy_delta_match": 0.0}
    t_logits, real_corpus = ref.generate_fixtures()
    syn_corpus = ref.sample_teacher(t_logits, 0, 100, 42)

    try:
        s_want = ref.fit_student_low_rank(t_logits, syn_corpus, 5)
        s_got = fit_student_low_rank(t_logits, syn_corpus, 5)
        if np.allclose(s_want, s_got, atol=1e-5):
            out["student_fit_match"] = 1.0

        r_want = ref.compare_distillation(t_logits, syn_corpus, real_corpus, 5)
        r_got = compare_distillation(t_logits, syn_corpus, real_corpus, 5)
        if np.allclose(r_want, r_got, atol=1e-5):
            out["accuracy_delta_match"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
