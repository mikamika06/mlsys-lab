def parse_fusion_trace(log_text):
    """
    Parse a string of TORCH_LOGS=fusion lines.
    Return a dict with keys:
      'fused_groups': list of lists of op names,
      'separate_ops': list of op names
    """
    raise NotImplementedError
