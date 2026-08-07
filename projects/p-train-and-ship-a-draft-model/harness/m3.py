def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    m = {"math_ok": 0.0, "eval_ok": 0.0}
    try:
        from spec.eval import expected_acceptance, average_acceptance
        from spec.draft import DraftModel

        p = np.array([0.1, 0.5, 0.4])
        q = np.array([0.2, 0.3, 0.5])
        acc = expected_acceptance(p, q)
        acc_ref = ref.oracle_expected_acceptance(p, q)

        if abs(acc - acc_ref) < 1e-5:
            m["math_ok"] = 1.0

        d = DraftModel(ref.get_vocab_size(), ref.get_hidden_size())
        target = ref.get_target_probs()
        dataset = ref.get_dataset()
        avg = average_acceptance(d, target, dataset)

        if 0.0 <= avg <= 1.0:
            m["eval_ok"] = 1.0
    except Exception:
        pass
    return m
