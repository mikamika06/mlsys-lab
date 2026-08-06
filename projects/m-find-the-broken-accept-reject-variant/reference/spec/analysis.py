import numpy as np


def compute_tv_distance(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


def compare_heuristics(p, q):
    tv_rs = compute_tv_distance(p, q)
    top1_p = np.zeros_like(p)
    top1_p[np.argmax(p)] = 1.0
    top1_q = np.zeros_like(q)
    top1_q[np.argmax(q)] = 1.0
    tv_top1 = float(0.5 * np.sum(np.abs(top1_p - top1_q)))
    return {"tv_rejection_sampling": tv_rs, "tv_top1": tv_top1}
