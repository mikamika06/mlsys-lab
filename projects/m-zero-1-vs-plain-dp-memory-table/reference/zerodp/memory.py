import math


def calc_memory_table(
    param_counts: list[int],
    world_size: int,
    bytes_per_param: int = 2,
    bytes_per_grad: int = 2,
    opt_bytes_per_param: int = 12,
) -> dict:
    """Calculate memory breakdown for Plain DP vs ZeRO Stage 1."""
    total_params = sum(param_counts)
    plain_p = total_params * bytes_per_param
    plain_g = total_params * bytes_per_grad
    plain_o = total_params * opt_bytes_per_param
    plain_tot = plain_p + plain_g + plain_o

    zero_p = plain_p
    zero_g = plain_g
    zero_o = math.ceil(plain_o / world_size) if world_size > 0 else 0
    zero_tot = zero_p + zero_g + zero_o

    return {
        "total_params": total_params,
        "plain_dp": {
            "params_bytes": plain_p,
            "grads_bytes": plain_g,
            "opt_bytes": plain_o,
            "total_bytes": plain_tot,
        },
        "zero1": {
            "params_bytes": zero_p,
            "grads_bytes": zero_g,
            "opt_bytes": zero_o,
            "total_bytes": zero_tot,
        },
        "opt_savings_bytes": plain_o - zero_o,
        "total_savings_bytes": plain_tot - zero_tot,
    }
