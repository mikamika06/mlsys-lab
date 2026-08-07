def alpha_beta_cost(message_size, world_size, alpha, beta, algorithm="ring"):
    if algorithm == "ring":
        return 2.0 * (world_size - 1) * alpha + ((world_size - 1) / world_size) * message_size * beta
    elif algorithm == "tree":
        return 2.0 * (world_size - 1) * alpha + 2.0 * ((world_size - 1) / world_size) * message_size * beta
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")


def find_crossover(world_size, alpha, beta, max_size=1048576):
    for size in range(1, max_size + 1):
        c_ring = alpha_beta_cost(size, world_size, alpha, beta, "ring")
        c_tree = alpha_beta_cost(size, world_size, alpha, beta, "tree")
        if c_ring <= c_tree:
            return size
    return max_size
