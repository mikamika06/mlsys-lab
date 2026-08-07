def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    m = {"loss_decreased": 0.0}
    try:
        from spec.draft import DraftModel
        from spec.train import train_draft

        d = DraftModel(ref.get_vocab_size(), ref.get_hidden_size())
        target = ref.get_target_probs()
        dataset = ref.get_dataset()

        def compute_kl(model):
            loss = 0.0
            for t in set(dataset):
                p = target[t]
                q = model.get_probs(t)
                loss += float(np.sum(p * np.log(p / (q + 1e-9))))
            return loss

        loss_before = compute_kl(d)
        d = train_draft(d, dataset, target, lr=0.5, epochs=10)
        loss_after = compute_kl(d)

        if loss_after < loss_before * 0.5:
            m["loss_decreased"] = 1.0
    except Exception:
        pass
    return m
