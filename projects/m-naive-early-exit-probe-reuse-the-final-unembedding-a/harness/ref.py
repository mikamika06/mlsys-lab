import numpy as np


def generate_fixture(seed=42):
    rng = np.random.default_rng(seed)
    hidden_states = rng.standard_normal((12, 48))
    unembed_weights = rng.standard_normal((128, 48))
    final_logits = rng.standard_normal((12, 128))
    hidden_dict = {
        3: rng.standard_normal((12, 48)),
        6: rng.standard_normal((12, 48)),
        9: rng.standard_normal((12, 48)),
    }
    published_table = {3: 0.35, 6: 0.65, 9: 0.90}
    return hidden_states, unembed_weights, final_logits, hidden_dict, published_table


def compute_early_exit_agreement(hidden_states, unembed_weights, final_logits):
    early_logits = np.dot(hidden_states, unembed_weights.T)
    early_preds = np.argmax(early_logits, axis=-1)
    final_preds = np.argmax(final_logits, axis=-1)
    return float(np.mean(early_preds == final_preds))


def sweep_and_compare(hidden_dict, unembed_weights, final_logits, published_table):
    results = {}
    diffs = []
    for layer_idx, hidden in sorted(hidden_dict.items()):
        score = compute_early_exit_agreement(hidden, unembed_weights, final_logits)
        results[layer_idx] = score
        if layer_idx in published_table:
            diffs.append(abs(score - published_table[layer_idx]))
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    return {"agreements": results, "mean_diff": mean_diff}
