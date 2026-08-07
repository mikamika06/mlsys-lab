import sys

sys.path.insert(0, ".")
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from memrunner.predictor import predict_resident_vram

    max_rel_err = 0.0
    for i, cfg in enumerate(ref.CONFIGS):
        seq_len = 1024 * (i + 1)
        batch_size = i + 1
        expected = ref.predict_resident_vram(cfg, seq_len=seq_len, batch_size=batch_size)
        got = predict_resident_vram(cfg, seq_len=seq_len, batch_size=batch_size)

        rel_err = abs(got - expected) / float(expected)
        if rel_err > max_rel_err:
            max_rel_err = rel_err

    return {"rel_err": float(max_rel_err)}
