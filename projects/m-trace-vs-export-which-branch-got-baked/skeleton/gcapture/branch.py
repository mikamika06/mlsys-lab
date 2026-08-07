def inspect_baked_branch(model, example_input, alternate_input):
    """
    Analyzes whether torch.jit.trace bakes a single branch path.
    Returns dict:
      'trace_baked_branch': bool,
      'trace_took_branch': str,
      'export_supports_alt': bool
    """
    raise NotImplementedError
