import ref
import numpy as np


def check(workdir):
    from bnb_quant.loader import load_model_weights, quantize_fp16_to_int8

    out = {"models_checked": 0.0, "load_accuracy": 0.0}
    models = ref.TEST_MODELS

    checked = 0
    correct = 0

    for model in models:
        checked += 1
        inflight_res = load_model_weights(model, mode="inflight")
        prequant_res = load_model_weights(model, mode="prequantized")

        model_ok = True
        for name in model:
            if name not in inflight_res or name not in prequant_res:
                model_ok = False
                break

            q1, s1 = inflight_res[name]["qweight"], inflight_res[name]["scales"]
            q2, s2 = prequant_res[name]["qweight"], prequant_res[name]["scales"]

            if not np.array_equal(q1, q2) or not np.allclose(s1, s2):
                model_ok = False
                break

        if model_ok:
            correct += 1

    out["models_checked"] = float(checked)
    out["load_accuracy"] = float(correct) / float(checked) if checked > 0 else 0.0
    return out
