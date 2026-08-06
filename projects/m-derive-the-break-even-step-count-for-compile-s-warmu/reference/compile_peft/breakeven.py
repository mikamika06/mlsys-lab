import math

def break_even_steps(config):
    c = config["warmup_cost"]
    te = config["eager_step"]
    tc = config["compiled_step"]
    saving = te - tc
    if saving <= 0:
        return float('inf')
    return math.ceil(c / saving)
