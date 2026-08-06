def analyze_fallback_causes(verbose_logs):
    """
    Parses a list of ONEDNN_VERBOSE log lines and extracts all fallback events.
    Returns a list of dicts:
    [
      {
        "primitive": str,
        "implementation": str,
        "reason": str, # 'unsupported_isa', 'unaligned_layout', or 'unsupported_datatype'
        "shape": str
      },
      ...
    ]
    """
    raise NotImplementedError
