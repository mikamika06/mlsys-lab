import numpy as np

def compute_perplexity(logits, target_ids):
    raise NotImplementedError

def compute_kl_divergence(p_logits, q_logits):
    raise NotImplementedError

def evaluate_model_quality(model_fn, dataset):
    raise NotImplementedError
