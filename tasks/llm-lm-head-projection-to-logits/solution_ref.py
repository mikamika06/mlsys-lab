def lm_head_projection(hidden_states: list[list[list[float]]],
                       weight: list[list[float]],
                       bias: list[float]) -> list[list[list[float]]]:
    """
    Compute logits from hidden states using LM head projection.
    Parameters
    ----------
    hidden_states : list[list[list[float]]]
        Shape (batch, seq_len, hidden_dim)
    weight : list[list[float]]
        Shape (vocab_size, hidden_dim)
    bias : list[float]
        Shape (vocab_size,)
    Returns
    -------
    logits : list[list[list[float]]]
        Shape (batch, seq_len, vocab_size)
    """
    batch = len(hidden_states)
    seq_len = len(hidden_states[0])
    hidden_dim = len(hidden_states[0][0])
    vocab_size = len(weight)

    logits = []
    for b in range(batch):
        batch_logits = []
        for s in range(seq_len):
            seq_logits = []
            for v in range(vocab_size):
                acc = 0.0
                for h in range(hidden_dim):
                    acc += float(hidden_states[b][s][h]) * float(weight[v][h])
                seq_logits.append(acc + float(bias[v]))
            batch_logits.append(seq_logits)
        logits.append(batch_logits)

    return logits
