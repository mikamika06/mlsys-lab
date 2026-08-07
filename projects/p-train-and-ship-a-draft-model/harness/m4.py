def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    m = {"is_int8": 0.0, "acc_maintained": 0.0}
    try:
        from spec.draft import DraftModel
        from spec.quant import QuantizedDraft
        from spec.train import train_draft
        from spec.eval import average_acceptance

        d = DraftModel(ref.get_vocab_size(), ref.get_hidden_size())
        target = ref.get_target_probs()
        dataset = ref.get_dataset()

        d = train_draft(d, dataset, target, lr=0.5, epochs=5)
        acc_f32 = average_acceptance(d, target, dataset)

        qd = QuantizedDraft(d)
        if qd.w1_q.dtype == np.int8 and qd.w2_q.dtype == np.int8:
            m["is_int8"] = 1.0

        acc_int8 = average_acceptance(qd, target, dataset)
        if abs(acc_f32 - acc_int8) < 0.1:
            m["acc_maintained"] = 1.0
    except Exception:
        pass
    return m
