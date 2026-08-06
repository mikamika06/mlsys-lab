"""Isolation module for GGUF failure analysis."""


def isolate_root_cause(sample):
    """Analyze a single metric diagnostic dict and return the failure cause string.

    Returns one of: 'tokenizer_damage', 'quantization_damage', 'engine_failure'.
    """
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
