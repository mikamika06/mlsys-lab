def get_oracle_estimates(num_params, world_size, act_mem, bytes_per_param=2, bytes_per_optim_state=12):
    w_mem = num_params * bytes_per_param
    g_mem = num_params * bytes_per_param
    o_mem = num_params * bytes_per_optim_state

    z1 = w_mem + g_mem + (o_mem / world_size) + act_mem
    z2 = w_mem + (g_mem / world_size) + (o_mem / world_size) + act_mem
    z3 = (w_mem / world_size) + (g_mem / world_size) + (o_mem / world_size) + act_mem

    psi = num_params * bytes_per_param
    scale = (world_size - 1) / world_size
    comm1_2 = 2.0 * psi * scale
    comm3 = 3.0 * psi * scale

    return {
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "comm1_2": comm1_2,
        "comm3": comm3
    }
