import sys


def frozenset_hash(values):
    width = sys.hash_info.width
    mask = (1 << width) - 1

    def shuffle_bits(h):
        return (((h ^ 89869747) ^ (h << 16)) * 3644798167) & mask

    unique = set(values)
    h = 0
    for value in unique:
        h ^= shuffle_bits(hash(value))

    h ^= ((len(unique) + 1) * 1927868237) & mask
    h ^= h >> 11
    h ^= h >> 25
    h = (h * 69069 + 907133923) & mask

    if h == mask:
        h = 590923713

    if h >= (1 << (width - 1)):
        h -= 1 << width

    return h
