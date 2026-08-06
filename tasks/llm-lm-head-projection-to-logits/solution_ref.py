import math
import numpy as np


def lm_head_projection(hidden_states: np.ndarray,
                       weight: np.ndarray,
                       bias: np.ndarray) -> np.ndarray:
    """
    Compute logits from hidden states using LM head projection.
    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (batch, seq_len, hidden_dim)
    weight : np.ndarray
        Shape (vocab_size, hidden_dim)
    bias : np.ndarray
        Shape (vocab_size,)
    Returns
    -------
    logits : np.ndarray
        Shape (batch, seq_len, vocab_size), dtype float64
    """
    batch, seq_len, hidden_dim = hidden_states.shape
    vocab_size = weight.shape[0]

    logits = np.empty((batch, seq_len, vocab_size), dtype=np.float64)

    for b in range(batch):
        for s in range(seq_len):
            for v in range(vocab_size):
                acc = 0.0
                for h in range(hidden_dim):
                    acc += float(hidden_states[b, s, h]) * float(weight[v, h])
                logits[b, s, v] = acc + float(bias[v])

    return logits
