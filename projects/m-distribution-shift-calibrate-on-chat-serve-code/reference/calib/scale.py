import numpy as np


def calibrate_scales(chat_stats, code_stats, base_scales):
    adjusted_scales = []
    adjusted_clips = []
    for c_stat, code_stat, scale in zip(chat_stats, code_stats, base_scales):
        ratio = (np.max(np.abs(code_stat)) + 1e-5) / (np.max(np.abs(c_stat)) + 1e-5)
        adj_scale = scale * np.clip(ratio, 0.8, 1.5)
        adj_clip = adj_scale * 0.95
        adjusted_scales.append(float(adj_scale))
        adjusted_clips.append(float(adj_clip))
    return adjusted_scales, adjusted_clips
