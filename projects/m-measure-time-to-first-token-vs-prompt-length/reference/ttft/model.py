import numpy as np


def fit_latency_model(prompt_lengths, ttfts):
    x = np.array(prompt_lengths, dtype=float)
    y = np.array(ttfts, dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"slope": float(m), "intercept": float(c)}


def predict_ttft(params, prompt_len):
    return float(params["slope"] * prompt_len + params["intercept"])
