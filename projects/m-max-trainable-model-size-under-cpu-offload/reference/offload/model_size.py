def max_trainable_model_size(gpu_mem_bytes, cpu_mem_bytes, seq_len, batch_size, hidden_dim, num_gpus):
    params_per_layer = 12 * hidden_dim * hidden_dim
    act_per_layer = batch_size * seq_len * hidden_dim * 34
    max_l = 0
    for l in range(1, 10000):
        total_params = l * params_per_layer
        gpu_param_mem = (2 * total_params) / num_gpus
        gpu_grad_mem = (2 * total_params) / num_gpus
        gpu_working_mem = 2 * params_per_layer
        gpu_act_mem = l * act_per_layer
        total_gpu = gpu_param_mem + gpu_grad_mem + gpu_working_mem + gpu_act_mem
        cpu_opt_mem = (16 * total_params) / num_gpus
        if total_gpu <= gpu_mem_bytes and cpu_opt_mem <= cpu_mem_bytes:
            max_l = l
        else:
            break
    return max_l * params_per_layer
