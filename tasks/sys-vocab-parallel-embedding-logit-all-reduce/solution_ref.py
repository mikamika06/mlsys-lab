import numpy as np


def vocab_parallel_forward(token_ids, embedding, output_weight, world_size):
    vocab, hidden = embedding.shape
    n = token_ids.shape[0]

    hidden_out = np.zeros((n, hidden), dtype=np.float64)
    logits = np.zeros((n, vocab), dtype=np.float64)

    bounds = np.linspace(0, vocab, world_size + 1, dtype=int)

    for rank in range(world_size):
        start = bounds[rank]
        end = bounds[rank + 1]

        local_hidden = np.zeros((n, hidden), dtype=np.float64)
        mask = (token_ids >= start) & (token_ids < end)
        local_hidden[mask] = embedding[token_ids[mask]]
        hidden_out += local_hidden

        local_logits = np.zeros((n, vocab), dtype=np.float64)
        local_logits[:, start:end] = hidden_out @ output_weight[start:end].T
        logits += local_logits

    return hidden_out, logits
