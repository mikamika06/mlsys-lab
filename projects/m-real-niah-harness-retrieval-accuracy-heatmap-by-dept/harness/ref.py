import numpy as np


def generate_task(length, depth_ratio, needle, haystack_filler="text"):
    tokens = [haystack_filler] * length
    pos = int(length * depth_ratio)
    tokens[pos] = needle
    return {"tokens": tokens, "depth_ratio": depth_ratio, "needle": needle, "pos": pos}


def score_heatmap(preds, truths, k=1):
    depths = sorted(list(preds.keys()))
    lengths = sorted(list(preds[depths[0]].keys()))
    matrix = np.zeros((len(depths), len(lengths)))
    for di, d in enumerate(depths):
        for li, l in enumerate(lengths):
            pred_list = preds[d][l]
            truth_list = truths[d][l]
            correct = sum(1 for p, t in zip(pred_list, truth_list) if t in p[:k])
            matrix[di, li] = correct / len(truth_list) if truth_list else 0.0
    return matrix
