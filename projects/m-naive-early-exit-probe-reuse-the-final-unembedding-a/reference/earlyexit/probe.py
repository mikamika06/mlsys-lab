import numpy as np


def compute_early_exit_agreement(hidden_states, unembed_weights, final_logits):
    early_logits = np.dot(hidden_states, unembed_weights.T)
    early_preds = np.argmax(early_logits, axis=-1)
    final_preds = np.argmax(final_logits, axis=-1)
    agreement = np.mean(early_preds == final_preds)
    return float(agreement)


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
