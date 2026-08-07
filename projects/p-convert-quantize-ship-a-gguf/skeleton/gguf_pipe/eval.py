def compute_perplexity(model_path, eval_data):
    raise NotImplementedError


def compute_kld(logits_ref, logits_quant):
    raise NotImplementedError


def measure_throughput(model_path, num_tokens=128):
    raise NotImplementedError
