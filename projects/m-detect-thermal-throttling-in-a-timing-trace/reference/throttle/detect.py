"""Throttling detection routines."""
import numpy as np


def extract_event_spans(is_throttled, min_consecutive=3):
    spans = []
    in_event = False
    start = 0
    count = 0

    for i, flag in enumerate(is_throttled):
        if flag:
            if not in_event:
                in_event = True
                start = i
                count = 1
            else:
                count += 1
        else:
            if in_event:
                if count >= min_consecutive:
                    spans.append((start, i - 1))
                in_event = False
                count = 0
    if in_event and count >= min_consecutive:
        spans.append((start, len(is_throttled) - 1))
    return spans


def detect_throttling_events(durations, window_size=5, threshold_factor=1.35, min_consecutive=3):
    arr = np.asarray(durations, dtype=np.float64)
    n = len(arr)
    if n < window_size:
        return []

    is_throttled = np.zeros(n, dtype=bool)
    for i in range(window_size, n):
        baseline = np.median(arr[i - window_size:i])
        if arr[i] >= baseline * threshold_factor:
            is_throttled[i] = True

    return extract_event_spans(is_throttled, min_consecutive=min_consecutive)
