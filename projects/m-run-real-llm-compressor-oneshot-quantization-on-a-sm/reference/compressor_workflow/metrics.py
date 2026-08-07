import ref

def measure_metrics(original_size, quantized_size, model_stub, dataset):
    ratio = ref.compute_compression_ratio(original_size, quantized_size)
    ppl = ref.compute_perplexity(model_stub, dataset)
    return {"compression_ratio": ratio, "perplexity": ppl}
