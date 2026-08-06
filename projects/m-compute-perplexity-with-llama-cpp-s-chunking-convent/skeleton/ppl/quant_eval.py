def dump_f16_logits(model, tokens, chunk_size, output_path):
    """Run model on token chunks and dump concatenated F16 logits to file."""
    raise NotImplementedError


def score_quantized_model(quant_model, tokens, chunk_size, f16_logits_path):
    """Score quantized model against reference F16 logits."""
    raise NotImplementedError
