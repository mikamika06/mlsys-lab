import numpy as np


def simulate_perplexity(seq_len, sink_size, window_size, strategy="sink_window"):
    np.random.seed(42)
    steps = np.arange(1, seq_len + 1, max(1, seq_len // 20))
    ppl = []
    for s in steps:
        if strategy == "full":
            val = 5.0 + 0.1 * np.sin(s / 100.0)
        elif strategy == "sink_window":
            if s <= sink_size + window_size:
                val = 5.0 + 0.2 * np.cos(s / 50.0)
            else:
                val = 5.2 + 0.05 * np.log(float(s))
        else:
            val = 20.0 + float(s) / 10.0
        ppl.append(float(val))
    return list(steps), ppl
