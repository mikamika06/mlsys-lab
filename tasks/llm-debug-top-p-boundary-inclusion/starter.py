def top_p_keep(probs, p):
    # TODO: boundary condition is wrong. This excludes a token when the
    # cumulative probability reaches p exactly.
    ordered = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept = []
    total = 0.0
    for i in ordered:
        total += probs[i]
        if total > p:
            break
        kept.append(i)
    return kept
