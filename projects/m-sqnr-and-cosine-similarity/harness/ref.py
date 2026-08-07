import numpy as np

np.random.seed(42)


def compute_sqnr(y_ref, y_test, eps=1e-12):
    y_ref = np.asarray(y_ref, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    signal_power = np.sum(y_ref ** 2)
    noise_power = np.sum((y_ref - y_test) ** 2)
    if noise_power < eps:
        return 120.0
    val = signal_power / noise_power
    if val <= 0:
        return -120.0
    return float(10.0 * np.log10(val))


def compute_cosine_similarity(y_ref, y_test, eps=1e-12):
    y_ref = np.asarray(y_ref, dtype=np.float64).flatten()
    y_test = np.asarray(y_test, dtype=np.float64).flatten()
    dot = np.dot(y_ref, y_test)
    norm_ref = np.linalg.norm(y_ref)
    norm_test = np.linalg.norm(y_test)
    denom = norm_ref * norm_test
    if denom < eps:
        return 0.0
    return float(dot / denom)


def compute_max_rel_err(y_ref, y_test, eps=1e-12):
    y_ref = np.asarray(y_ref, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    diff = np.abs(y_ref - y_test)
    denom = np.abs(y_ref) + eps
    return float(np.max(diff / denom))


def evaluate_gate(y_ref, y_test, min_sqnr_db=30.0, min_cos_sim=0.99, max_rel_err=1e-2, eps=1e-12):
    sqnr_val = compute_sqnr(y_ref, y_test, eps=eps)
    cos_val = compute_cosine_similarity(y_ref, y_test, eps=eps)
    rel_err_val = compute_max_rel_err(y_ref, y_test, eps=eps)
    passed = bool((sqnr_val >= min_sqnr_db) and (cos_val >= min_cos_sim) and (rel_err_val <= max_rel_err))
    return {
        "passed": passed,
        "sqnr_db": sqnr_val,
        "cos_sim": cos_val,
        "max_rel_err": rel_err_val,
    }


def analyze_amplification(layer_refs, layer_tests, eps=1e-12):
    layer_errors = []
    for r, t in zip(layer_refs, layer_tests):
        r_arr = np.asarray(r, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        norm_ref = np.linalg.norm(r_arr)
        diff_norm = np.linalg.norm(r_arr - t_arr)
        err = float(diff_norm / (norm_ref + eps))
        layer_errors.append(err)

    amplifications = [1.0]
    for i in range(1, len(layer_errors)):
        prev_err = layer_errors[i - 1]
        curr_err = layer_errors[i]
        amp = float(curr_err / (prev_err + eps))
        amplifications.append(amp)

    max_idx = int(np.argmax(amplifications))
    return {
        "layer_errors": layer_errors,
        "amplifications": amplifications,
        "max_amplifying_layer": max_idx,
    }


METRIC_TEST_CASES = []
for seed in range(5):
    rng = np.random.RandomState(seed)
    ref_vec = rng.randn(1000)
    noise = rng.randn(1000) * (0.01 * (seed + 1))
    test_vec = ref_vec + noise
    METRIC_TEST_CASES.append((ref_vec, test_vec))

GATE_TEST_CASES = []
for seed in range(5):
    rng = np.random.RandomState(10 + seed)
    ref_vec = rng.randn(500)
    noise_scale = 0.001 * (10 ** (seed * 0.8))
    test_vec = ref_vec + rng.randn(500) * noise_scale
    GATE_TEST_CASES.append((ref_vec, test_vec, 25.0 + seed * 2.0, 0.95 + seed * 0.01, 0.05 / (seed + 1)))

LAYER_TEST_CASES = []
for seed in range(3):
    rng = np.random.RandomState(20 + seed)
    num_layers = 5
    ref_chain = []
    test_chain = []
    cur_ref = rng.randn(64, 64)
    cur_test = cur_ref + rng.randn(64, 64) * 0.001
    for i in range(num_layers):
        W = rng.randn(64, 64) / np.sqrt(64)
        cur_ref = np.maximum(0, np.dot(cur_ref, W))
        if i == 2:
            cur_test = np.maximum(0, np.dot(cur_test, W)) + rng.randn(64, 64) * 0.1
        else:
            cur_test = np.maximum(0, np.dot(cur_test, W)) + rng.randn(64, 64) * 0.0005
        ref_chain.append(cur_ref)
        test_chain.append(cur_test)
    LAYER_TEST_CASES.append((ref_chain, test_chain))
