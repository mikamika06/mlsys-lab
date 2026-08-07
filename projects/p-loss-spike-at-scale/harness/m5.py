def check(workdir):
    import ref
    import numpy as np
    m = {"train_matches": 0.0}
    try:
        from system import train, distributed
        tensors = ref.get_test_tensors()
        norm = train.train_step(tensors, distributed.safe_all_reduce_sum)
        expected = float(np.sum(sum(tensors)**2))

        if abs(norm - expected) < 1e-3:
            m["train_matches"] = 1.0
    except Exception:
        pass
    return m
