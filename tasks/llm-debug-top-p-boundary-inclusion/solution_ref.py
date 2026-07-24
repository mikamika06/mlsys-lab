def top_p_keep(probs, p):
    ordered = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept = []
    total = 0.0
    for i in ordered:
        kept.append(i)
        total += probs[i]
        if total >= p:
            break
    return kept
