import math


def choose_block_size(M, N, D):
    best = None
    best_pair = (1, 1)
    for br in range(1, N + 1):
        for bc in range(1, N + 1):
            if br * D + bc * D + br * bc <= M:
                traffic = (
                    2 * N * D
                    + 2 * math.ceil(N / br) * math.ceil(N / bc) * bc * D
                )
                if best is None or traffic < best:
                    best = traffic
                    best_pair = (br, bc)
    return best_pair
