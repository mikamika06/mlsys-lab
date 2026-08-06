def parse_llama_bench_json(raw_json_str):
    """Parse raw llama-bench JSON output into a normalized record dictionary."""
    raise NotImplementedError


def extract_quant_metrics(parsed_data, quant_type):
    """Extract pp and tg tokens/sec for a specific quantization type."""
    raise NotImplementedError
