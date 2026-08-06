import numpy as np
import math

def entropy_calibration_threshold(hist: np.ndarray) -> tuple[int, np.ndarray]:
    hist_len = len(hist)
    hist_val = np.zeros(hist_len, dtype=np.float64)
    for i in range(hist_len):
        hist_val[i] = float(hist[i])

    values = []

    for t in range(128, 2048):
        clip_len = t + 1 if t + 1 < hist_len else hist_len
        clipped = np.zeros(clip_len, dtype=np.float64)
        for i in range(clip_len):
            clipped[i] = hist_val[i]

        if t + 1 < hist_len:
            tail_sum = 0.0
            for i in range(t + 1, hist_len):
                tail_sum += hist_val[i]
            clipped[-1] += tail_sum

        clip_sum = 0.0
        for i in range(clip_len):
            clip_sum += clipped[i]

        p = np.zeros(clip_len, dtype=np.float64)
        for i in range(clip_len):
            p[i] = clipped[i] / clip_sum

        q_counts = np.zeros(clip_len, dtype=np.float64)
        edges = np.zeros(129, dtype=np.int64)
        step = (t + 1) / 128.0
        for i in range(128):
            edges[i] = int(i * step)
        edges[128] = t + 1

        for i in range(128):
            start = edges[i]
            end = edges[i + 1]
            if end > start:
                actual_start = start if start < clip_len else clip_len
                actual_end = end if end < clip_len else clip_len
                
                range_sum = 0.0
                for j in range(actual_start, actual_end):
                    range_sum += clipped[j]
                
                val = range_sum / (end - start)
                for j in range(actual_start, actual_end):
                    q_counts[j] = val

        q_sum = 0.0
        for i in range(clip_len):
            q_sum += q_counts[i]

        q = np.zeros(clip_len, dtype=np.float64)
        for i in range(clip_len):
            q[i] = q_counts[i] / q_sum

        kl_sum = 0.0
        for i in range(clip_len):
            if p[i] > 0.0:
                kl_sum += p[i] * math.log((p[i] + 1e-12) / (q[i] + 1e-12))
        
        values.append(kl_sum)

    curve_len = len(values)
    curve = np.zeros(curve_len, dtype=np.float64)
    min_val = float('inf')
    min_idx = 0
    
    for i in range(curve_len):
        curve[i] = values[i]
        if curve[i] < min_val:
            min_val = curve[i]
            min_idx = i

    return int(min_idx + 128), curve
