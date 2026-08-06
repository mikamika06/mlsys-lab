from preemption.crossover import find_crossover

def choose_cheaper_mode(context_len, model_config, system_config):
    crossover_blocks = find_crossover(model_config, system_config)
    blocks = (context_len + system_config["block_size"] - 1) // system_config["block_size"]
    return "swap" if blocks < crossover_blocks else "recompute"
