"""Diagnosis of autoscaler thrashing."""

def detect_thrashing(trace, threshold=3):
    counts = [r[1] for r in trace]
    flips = 0
    for i in range(1, len(counts) - 1):
        if (counts[i] > counts[i-1] and counts[i] > counts[i+1]) or (counts[i] < counts[i-1] and counts[i] < counts[i+1]):
            flips += 1
    return flips >= threshold
