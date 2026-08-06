def predict_ready_time(pull_size, weight_size, compile_factor, pull_speed, weight_speed):
    t_pull = pull_size / pull_speed
    t_weight = weight_size / weight_speed
    t_compile = (pull_size + weight_size) * compile_factor / 1000.0
    return float(t_pull + t_weight + t_compile)
