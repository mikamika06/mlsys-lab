"""Reference oracle data generator and baseline implementation."""

import random


def generate_samples(seed=42, n=100):
    rng = random.Random(seed)
    causes = ["tokenizer_damage", "quantization_damage", "engine_failure"]
    samples = []
    labels = []

    for _ in range(n):
        cause = rng.choice(causes)
        labels.append(cause)
        if cause == "engine_failure":
            sample = {
                "engine_panic": rng.random() < 0.6,
                "buffer_overflow": rng.random() < 0.5,
                "context_index_error": rng.random() < 0.4,
                "unk_token_ratio": rng.uniform(0.0, 0.05),
                "bos_eos_missing": False,
                "id_out_of_bounds": False,
                "ppl_spike": rng.uniform(0.0, 10.0),
                "has_nan_inf": False,
                "logit_kl_divergence": rng.uniform(0.0, 0.5),
            }
            if not (sample["engine_panic"] or sample["buffer_overflow"] or sample["context_index_error"]):
                sample["engine_panic"] = True
        elif cause == "tokenizer_damage":
            sample = {
                "engine_panic": False,
                "buffer_overflow": False,
                "context_index_error": False,
                "unk_token_ratio": rng.uniform(0.18, 0.8),
                "bos_eos_missing": rng.random() < 0.5,
                "id_out_of_bounds": rng.random() < 0.3,
                "ppl_spike": rng.uniform(0.0, 5.0),
                "has_nan_inf": False,
                "logit_kl_divergence": rng.uniform(0.0, 0.8),
            }
        else:
            sample = {
                "engine_panic": False,
                "buffer_overflow": False,
                "context_index_error": False,
                "unk_token_ratio": rng.uniform(0.0, 0.05),
                "bos_eos_missing": False,
                "id_out_of_bounds": False,
                "ppl_spike": rng.uniform(55.0, 500.0),
                "has_nan_inf": rng.random() < 0.7,
                "logit_kl_divergence": rng.uniform(2.6, 10.0),
            }
        samples.append(sample)

    return samples, labels


def reference_isolate(sample):
    if sample.get("engine_panic", False) or sample.get("buffer_overflow", False) or sample.get("context_index_error", False):
        return "engine_failure"

    tok_unk_ratio = sample.get("unk_token_ratio", 0.0)
    tok_bos_eos_missing = sample.get("bos_eos_missing", False)
    tok_id_out_of_bounds = sample.get("id_out_of_bounds", False)

    if tok_unk_ratio > 0.15 or tok_bos_eos_missing or tok_id_out_of_bounds:
        return "tokenizer_damage"

    quant_ppl_spike = sample.get("ppl_spike", 0.0)
    quant_has_nan_inf = sample.get("has_nan_inf", False)
    quant_logit_kl_div = sample.get("logit_kl_divergence", 0.0)

    if quant_has_nan_inf or quant_ppl_spike > 50.0 or quant_logit_kl_div > 2.5:
        return "quantization_damage"

    return "tokenizer_damage"
