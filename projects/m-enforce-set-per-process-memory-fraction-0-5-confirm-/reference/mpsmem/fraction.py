def enforce_fraction(fraction, capacity):
    return int(capacity * fraction)


def check_oom(limit, allocated, requested):
    return (allocated + requested) > limit
