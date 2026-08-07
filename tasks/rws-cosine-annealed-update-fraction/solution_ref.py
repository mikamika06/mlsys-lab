import math

def cosine_annealed_update_fraction(f0, T, t, nnz):
    ft = f0 / 2 * (1 + math.cos(math.pi * t / T))
    return round(ft * nnz)
