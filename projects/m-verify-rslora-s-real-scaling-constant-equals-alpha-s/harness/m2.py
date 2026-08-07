import ref
import numpy as np

def check(workdir):
    from adapters.compare import compare_magnitudes, compare_parameters
    cases = ref.generate_test_cases()
    out = {"outputs_compared": 0.0, "params_compared": 0.0}

    mag_ok = True
    for case in cases:
        x = case["x"]
        w_a = case["w_a"]
        w_b = case["w_b"]
        r = case["rank"]
        alpha = case["alpha"]

        lora_out = np.dot(np.dot(x, w_a.T), w_b.T) * (alpha / r)
        rslora_out = np.dot(np.dot(x, w_a.T), w_b.T) * (alpha / np.sqrt(r))

        got_lora, got_rslora = compare_magnitudes(x, w_a, w_b, alpha, r)
        if not (np.allclose(got_lora, lora_out, atol=1e-4) and np.allclose(got_rslora, rslora_out, atol=1e-4)):
            mag_ok = False
            break

    if mag_ok:
        out["outputs_compared"] = 1.0

    # Check parameters
    # IA3 vs LoRA parameter totals for a mock layer shape (in_features=64, out_features=64)
    p_lora, p_ia3 = compare_parameters(in_features=64, out_features=64, rank=8)
    expected_lora = 2 * 8 * 64
    expected_ia3 = 64
    if p_lora == expected_lora and p_ia3 == expected_ia3:
        out["params_compared"] = 1.0
    else:
        out["_note"] = f"Parameter count mismatch: got lora={p_lora}, ia3={p_ia3}"

    return out
