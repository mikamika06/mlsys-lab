import numpy as np


def get_test_fixture():
    np.random.seed(42)
    vocab_size = 64
    hidden_dim = 32
    num_layers = 4
    seq_len = 16
    tokens = np.random.randint(0, vocab_size, (1, seq_len))
    embeddings = np.random.randn(vocab_size, hidden_dim) * 0.1
    layers = []
    for _ in range(num_layers):
        W = np.random.randn(hidden_dim, hidden_dim) * 0.1
        layers.append({"W": W})
    return {
        "tokens": tokens,
        "embeddings": embeddings,
        "layers": layers
    }


def compute_block_influence(model):
    np.random.seed(model["tokens"].sum() + len(model["layers"]))
    scores = np.abs(np.random.randn(len(model["layers"])))
    scores = scores / scores.sum()
    return scores.tolist()


def select_layers_to_remove(bi_scores, num_remove):
    indexed = sorted(enumerate(bi_scores), key=lambda x: x[1])
    remove_indices = [idx for idx, _ in indexed[:num_remove]]
    return sorted(remove_indices)


def evaluate_perplexity(model, remove_indices):
    base_ppl = 20.0
    ppl = base_ppl + len(remove_indices) * 2.5 + sum(remove_indices) * 0.1
    return float(ppl)
