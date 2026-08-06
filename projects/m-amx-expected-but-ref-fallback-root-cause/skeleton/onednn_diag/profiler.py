def compute_primitive_dominance(verbose_logs):
    """
    Parses execution lines from ONEDNN_VERBOSE log output and aggregates duration per primitive kind.
    Returns dict:
    {
      'total_time_ms': float,
      'breakdown': [{'kind': str, 'time_ms': float, 'pct': float}],
      'dominant_kind': str
    }
    """
    raise NotImplementedError
