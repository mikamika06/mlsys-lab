import numpy as np
from typing import List, Dict, Any


def quantify_ttft_side_channel(ttft_hits: List[float], ttft_misses: List[float]) -> Dict[str, float]:
    hits = np.array(ttft_hits, dtype=np.float64)
    misses = np.array(ttft_misses, dtype=np.float64)

    mean_hit = float(np.mean(hits))
    mean_miss = float(np.mean(misses))
    delta_ttft = float(mean_miss - mean_hit)

    std_hit = float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0
    std_miss = float(np.std(misses, ddof=1)) if len(misses) > 1 else 0.0

    n_hit = len(hits)
    n_miss = len(misses)
    pooled_std = float(np.sqrt(((n_hit - 1) * (std_hit ** 2) + (n_miss - 1) * (std_miss ** 2)) / (n_hit + n_miss - 2)))

    cohens_d = float(delta_ttft / pooled_std) if pooled_std > 0 else 0.0
    p95_hit = float(np.percentile(hits, 95))
    p5_miss = float(np.percentile(misses, 5))

    separable = float(p95_hit < p5_miss)

    return {
        "mean_hit_ms": mean_hit,
        "mean_miss_ms": mean_miss,
        "delta_ttft_ms": delta_ttft,
        "cohens_d": cohens_d,
        "distinguishable": separable
    }
