import ref
import numpy as np

def check(workdir):
    from gptqquant.config import make_config
    from gptqquant.quantize import quantize_weights
    from gptqquant.export import calculate_size_ratio

    out = {"size_ratio_match": 0.0, "quantized_correctly": 0.0}
    w = ref.get_test_weight()
    cfg = make_config(bits=4, group_size=64, sym=True)

    try:
        ref_qw, ref_s, ref_z = quantize_weights(w, cfg)
        ref_ratio = calculate_size_ratio(w, ref_qw, ref_s, ref_z, cfg)

        from gptqquant.quantize import quantize_weights as learner_qw_fn
        from gptqquant.export import calculate_size_ratio as learner_ratio_fn

        l_qw, l_s, l_z = learner_qw_fn(w, cfg)
        l_ratio = learner_ratio_fn(w, l_qw, l_s, l_z, cfg)

        if l_qw.shape == ref_qw.shape and np.allclose(l_qw, ref_qw, atol=1e-5):
            out["quantized_correctly"] = 1.0

        if np.isclose(l_ratio, ref_ratio, atol=1e-4):
            out["size_ratio_match"] = 1.0
        else:
            out["_note"] = f"size ratio mismatch: got {l_ratio}, expected {ref_ratio}"
    except Exception as e:
        out["_note"] = f"m2 error: {type(e).__name__}: {str(e)[:120]}"

    return out
