from .histogram import build_histogram


def repetition_report(tokens, window, threshold):
    tokens = list(tokens)
    tail = tokens[-window:] if window > 0 else []
    tail_hist = build_histogram(tail)
    full_hist = build_histogram(tokens)

    triggered = False
    token = None
    window_count = 0
    if tail_hist:
        best = max(tail_hist.values())
        if best >= threshold:
            token = min(t for t, c in tail_hist.items() if c == best)
            window_count = best
            triggered = True

    positions = sorted(i for i, t in enumerate(tokens) if t == token) if triggered else []

    return {
        "triggered": triggered,
        "token": token,
        "window_count": window_count,
        "positions": positions,
        "histogram": full_hist,
        "total_tokens": len(tokens),
        "unique_tokens": len(full_hist),
    }
