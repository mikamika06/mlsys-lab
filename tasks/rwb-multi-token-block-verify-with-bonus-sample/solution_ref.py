def _sample(dist, u):
    total = sum(dist)
    if total <= 0:
        return 0
    cumulative = 0.0
    for token, prob in enumerate(dist):
        cumulative += prob / total
        if u < cumulative:
            return token
    return len(dist) - 1


def verify_block(draft, p, q, rng):
    accepted = 0
    rng_index = 0
    rejected = False

    for i, token in enumerate(draft):
        accept_prob = 0.0
        if q[i][token] > 0:
            accept_prob = min(1.0, p[i][token] / q[i][token])
        if rng[rng_index] < accept_prob:
            accepted += 1
            rng_index += 1
        else:
            rng_index += 1
            rejected = True
            break

    if rejected:
        residual = [max(p[accepted][j] - q[accepted][j], 0.0) for j in range(len(p[accepted]))]
        bonus = _sample(residual, rng[rng_index])
    else:
        bonus = _sample(p[-1], rng[rng_index])

    return accepted, draft[:accepted] + [bonus]
