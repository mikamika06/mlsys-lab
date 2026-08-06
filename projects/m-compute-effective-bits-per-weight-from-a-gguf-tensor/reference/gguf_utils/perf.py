import numpy as np


def fit_decode_performance(quants_data):
    bpw = np.array([q["bpw"] for q in quants_data], dtype=float)
    toks = np.array([q["tok_s"] for q in quants_data], dtype=float)
    slope, intercept = np.polyfit(bpw, toks, 1)
    return {"slope": float(slope), "intercept": float(intercept)}
