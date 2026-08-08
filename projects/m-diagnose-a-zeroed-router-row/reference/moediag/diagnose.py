from moediag.params import count_parameters
from moediag.router import find_zeroed_rows


def diagnose_moe(router_weights, config):
    zeroed = find_zeroed_rows(router_weights)
    params = count_parameters(config)
    return {
        "zeroed_rows": zeroed,
        "total_parameters": params["total_parameters"],
        "active_parameters": params["active_parameters"],
    }
