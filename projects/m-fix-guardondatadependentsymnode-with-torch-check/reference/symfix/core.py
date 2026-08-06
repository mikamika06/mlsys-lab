def apply_check(cfg):
    return f"torch._check(x.shape[0] <= {cfg['max_limit']}); return x * {cfg['input_dim']}"
