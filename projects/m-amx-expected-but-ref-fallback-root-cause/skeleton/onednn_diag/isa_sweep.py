def analyze_k_sweep(sweep_records):
    """
    Analyzes kernel selection over a list of records:
    [{'k': int, 'isa': str, 'latency_ms': float}, ...]
    Returns dict:
    {
      'transitions': [{'from_isa': str, 'to_isa': str, 'at_k': int}],
      'dominant_isa': str,
      'amx_efficiency_gain': float
    }
    """
    raise NotImplementedError
